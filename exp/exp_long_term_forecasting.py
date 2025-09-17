from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.result_path = args.result_path
        self.ps_lambda = args.ps_lambda
        self.use_ps_loss = args.use_ps_loss
        self.patch_len_threshold = args.patch_len_threshold
        self.kl_loss = nn.KLDivLoss(reduction='none')

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def create_patches(self, x, patch_len, stride):

        x = x.permute(0, 2, 1)  # [B, C, L] -> [B, L, C]
        B, C, L = x.shape

        num_patches = (L - patch_len) // stride + 1
        patches = x.unfold(2, patch_len, stride)
        patches = patches.reshape(B, C, num_patches, patch_len)

        return patches

    def fouriour_based_adaptive_patching(self, true, pred):

        # Get patch length an stride
        true_fft = torch.fft.rfft(true, dim=1)
        frequency_list = torch.abs(true_fft).mean(0).mean(-1)
        frequency_list[:1] = 0.0
        top_index = torch.argmax(frequency_list)
        period = (true.shape[1] // top_index)
        patch_len = min(period // 2, self.patch_len_threshold)
        stride = patch_len // 2

        # Patching
        true_patch = self.create_patches(true, patch_len, stride=stride)
        pred_patch = self.create_patches(pred, patch_len, stride=stride)

        return true_patch, pred_patch

    def patch_wise_structural_loss(self, true_patch, pred_patch):

        # Calculate mean
        true_patch_mean = torch.mean(true_patch, dim=-1, keepdim=True)
        pred_patch_mean = torch.mean(pred_patch, dim=-1, keepdim=True)

        # Calculate variance and standard deviation
        true_patch_var = torch.var(true_patch, dim=-1, keepdim=True, unbiased=False)
        pred_patch_var = torch.var(pred_patch, dim=-1, keepdim=True, unbiased=False)
        true_patch_std = torch.sqrt(true_patch_var)
        pred_patch_std = torch.sqrt(pred_patch_var)

        # Calculate Covariance
        true_pred_patch_cov = torch.mean((true_patch - true_patch_mean) * (pred_patch - pred_patch_mean), dim=-1,
                                         keepdim=True)

        # 1. Calculate linear correlation loss
        patch_linear_corr = (true_pred_patch_cov + 1e-5) / (true_patch_std * pred_patch_std + 1e-5)
        linear_corr_loss = (1.0 - patch_linear_corr).mean()

        # 2. Calculate variance
        true_patch_softmax = torch.softmax(true_patch, dim=-1)
        pred_patch_softmax = torch.log_softmax(pred_patch, dim=-1)
        var_loss = self.kl_loss(pred_patch_softmax, true_patch_softmax).sum(dim=-1).mean()

        # 3. Mean loss
        mean_loss = torch.abs(true_patch_mean - pred_patch_mean).mean()

        return linear_corr_loss, var_loss, mean_loss

    def ps_loss(self, true, pred):

        # Fourior based adaptive patching
        true_patch, pred_patch = self.fouriour_based_adaptive_patching(true, pred)

        # Pacth-wise structural loss
        corr_loss, var_loss, mean_loss = self.patch_wise_structural_loss(true_patch, pred_patch)

        # Gradient based dynamic weighting
        alpha, beta, gamma = self.gradient_based_dynamic_weighting(true, pred, corr_loss, var_loss, mean_loss)

        # Final PS loss
        ps_loss = alpha * corr_loss + beta * var_loss + gamma * mean_loss

        return ps_loss

    def gradient_based_dynamic_weighting(self, true, pred, corr_loss, var_loss, mean_loss):

        true = true.permute(0, 2, 1)
        pred = pred.permute(0, 2, 1)
        true_mean = torch.mean(true, dim=-1, keepdim=True)
        pred_mean = torch.mean(pred, dim=-1, keepdim=True)
        true_var = torch.var(true, dim=-1, keepdim=True, unbiased=False)
        pred_var = torch.var(pred, dim=-1, keepdim=True, unbiased=False)
        true_std = torch.sqrt(true_var)
        pred_std = torch.sqrt(pred_var)
        true_pred_cov = torch.mean((true - true_mean) * (pred - pred_mean), dim=-1, keepdim=True)
        linear_sim = (true_pred_cov + 1e-5) / (true_std * pred_std + 1e-5)
        linear_sim = (1.0 + linear_sim) * 0.5
        var_sim = (2 * true_std * pred_std + 1e-5) / (true_var + pred_var + 1e-5)

        # Gradiant based dynamic weighting
        # AMPTST, Patch_TST
        # if self.args.head_or_projection == 0:
        #     corr_gradient = torch.autograd.grad(corr_loss, self.model.head.parameters(), create_graph=True)[0]
        #     var_gradient = torch.autograd.grad(var_loss, self.model.head.parameters(), create_graph=True)[0]
        #     mean_gradient = torch.autograd.grad(mean_loss, self.model.head.parameters(), create_graph=True)[0]
        # else:
        #     corr_gradient = torch.autograd.grad(corr_loss, self.model.projection.parameters(), create_graph=True)[0]
        #     var_gradient = torch.autograd.grad(var_loss, self.model.projection.parameters(), create_graph=True)[0]
        #     mean_gradient = torch.autograd.grad(mean_loss, self.model.projection.parameters(), create_graph=True)[0]
        # TimeMixer
        # corr_gradient = torch.autograd.grad(corr_loss, self.model.predict_layers[-1].parameters(), create_graph=True)[0]
        # var_gradient = torch.autograd.grad(var_loss, self.model.predict_layers[-1].parameters(), create_graph=True)[0]
        # mean_gradient = torch.autograd.grad(mean_loss, self.model.predict_layers[-1].parameters(), create_graph=True)[0]
        # TimesNet iTransformer
        corr_gradient = torch.autograd.grad(corr_loss, self.model.projection.parameters(), create_graph=True)[0]
        var_gradient = torch.autograd.grad(var_loss, self.model.projection.parameters(), create_graph=True)[0]
        mean_gradient = torch.autograd.grad(mean_loss, self.model.projection.parameters(), create_graph=True)[0]
        # DLinear
        # corr_gradient = torch.autograd.grad(corr_loss, self.model.Linear_Trend.parameters(), create_graph=True)[0]
        # var_gradient = torch.autograd.grad(var_loss, self.model.Linear_Trend.parameters(), create_graph=True)[0]
        # mean_gradient = torch.autograd.grad(mean_loss, self.model.Linear_Trend.parameters(), create_graph=True)[0]
        gradiant_avg = (corr_gradient + var_gradient + mean_gradient) / 3.0

        aplha = gradiant_avg.norm().detach() / corr_gradient.norm().detach()
        beta = gradiant_avg.norm().detach() / var_gradient.norm().detach()
        gamma = gradiant_avg.norm().detach() / mean_gradient.norm().detach()
        gamma = gamma * torch.mean(linear_sim * var_sim).detach()

        return aplha, beta, gamma

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)

                    if self.use_ps_loss:
                        ps_loss = self.ps_loss(batch_y, outputs)
                        loss += ps_loss * self.ps_lambda

                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        if test_data.scale and self.args.inverse:
            preds_inverse = []
            trues_inverse = []
        folder_path = './test_results/' + setting + '/' + self.result_path + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs_inverse = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y_inverse = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    outputs_inverse = outputs_inverse[:, :, f_dim:]
                    batch_y_inverse = batch_y_inverse[:, :, f_dim:]
                    pi = outputs_inverse
                    ti = batch_y_inverse
                    preds_inverse.append(pi)
                    trues_inverse.append(ti)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    visual(true[0, :, -1], pred[0, :, -1], os.path.join(folder_path, str(i) + 'perd.pdf'))
                    input = batch_x.detach().cpu().numpy()
                    # if test_data.scale and self.args.inverse:
                        # shape = input.shape
                        # input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                        # visual(np.array(trues_inverse)[0, :, -1],np.array(preds_inverse)[0, :, -1], os.path.join(folder_path, str(i) + 'perd_inverse.pdf'))
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        if test_data.scale and self.args.inverse:
            preds_inverse = np.concatenate(preds_inverse, axis=0)
            trues_inverse = np.concatenate(trues_inverse, axis=0)
            preds_inverse = preds_inverse.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues_inverse = trues_inverse.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe, r2 = metric(preds, trues)
        mae_i, mse_i, rmse_i, mape_i, mspe_i, r2_i = metric(preds_inverse, trues_inverse)
        print('mse:{}, mae:{}, mape_i:{}'.format(mse, mae, mape_i))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, mape_i:{}'.format(mse, mae, mape_i))
        f.write('\n')
        f.write('\n')
        f.close()
        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape_i, mspe]))
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, r2, mae_i, mse_i, rmse_i, mape_i, mspe_i, r2_i]))
        if self.args.storage_result:
            np.save(folder_path + 'pred.npy', preds_inverse)
            np.save(folder_path + 'true.npy', trues_inverse)
        return
