---------------------------------------------------------------------------------------------
create database AMPTST_result;
USE AMPTST_result;

CREATE TABLE IF NOT EXISTS exp_results_t2 (
    id            BIGINT AUTO_INCREMENT PRIMARY KEY,
    dataset             INT          COMMENT 'history length',
    sl            INT          COMMENT 'sequence length',
    pl            INT          COMMENT 'pred length',
    model         VARCHAR(64)  COMMENT 'AMPTST / etc.',
    ft            VARCHAR(16)  COMMENT 'feature type',
    ll            INT          COMMENT 'label length',
    dm            INT          COMMENT 'd_model',
    nh            INT          COMMENT 'n_heads',
    el            INT          COMMENT 'e_layers',
    dl            INT          COMMENT 'd_layers',
    df            INT          COMMENT 'd_ff',
    fc            INT          COMMENT 'factor',
    ebtime        VARCHAR(8)   COMMENT 'embed time',
    dt            BOOLEAN      COMMENT 'distil',
    describe_     VARCHAR(64),
    mse           DOUBLE       NOT NULL,
    mae           DOUBLE       NOT NULL,
    mape_i        DOUBLE       NOT NULL,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uniq_run (dataset,sl,pl,model,ft,ll,dm,nh,el,dl,df,fc,ebtime,dt,describe_)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS exp_results_v3 (
    id        BIGINT AUTO_INCREMENT PRIMARY KEY,
    dataset   VARCHAR(20)  COMMENT '数据集代号',
    sl        INT          COMMENT 'sequence length',
    pl        INT          COMMENT 'predict length',
    model     VARCHAR(20)  COMMENT '模型名',
    ft        VARCHAR(10)  COMMENT 'feature type',
    ll        INT          COMMENT 'label length',
    dm        INT          COMMENT 'd_model',
    nh        INT          COMMENT 'n_heads',
    el        INT          COMMENT 'e_layers',
    dl        INT          COMMENT 'd_layers',
    df        INT          COMMENT 'd_ff',
    fc        INT          COMMENT 'factor',
    ebtime    VARCHAR(8)   COMMENT 'embed time',
    dt        BOOLEAN      COMMENT 'distil',
    describe_ VARCHAR(64)  COMMENT '实验描述',
    mse       DOUBLE       NOT NULL,
    mae       DOUBLE       NOT NULL,
    mape_i    DOUBLE       NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uniq_run (dataset,sl,pl,model,ft,ll,dm,nh,el,dl,df,fc,ebtime,dt,describe_)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

SELECT *
FROM exp_results_t2
ORDER BY mse
LIMIT 1;

SELECT
    dataset,sl,pl,model, mse, mae, mape_i,describe_
FROM exp_results_t4
WHERE sl = 168 AND describe_ IN ('CM', 'Exp')
order by model,pl,mae;

SELECT
    REGEXP_SUBSTR(describe_, '[0-9]+\\.[0-9]+') AS val,
    dataset,sl,pl,model, mse, mae, mape_i,describe_
FROM exp_results_t4
WHERE sl = 168
  AND describe_ REGEXP '^(CM|pw)(2\\.0|6\\.0|10\\.0)$'
order by model,pl,mse,mae,mape_i;

SELECT
    REGEXP_SUBSTR(describe_, '[0-9]+\\.[0-9]+') AS val,
    dataset,sl,pl,model, mse, mae, mape_i,describe_
FROM exp_results_t4
WHERE sl = 96
  AND describe_ REGEXP '^(CM|pw)(2\\.0)$'
order by val,pl,mape_i;

SELECT dataset,sl,pl,describe_,model, mse, mae, mape_i
FROM exp_results_t4
WHERE model = 'AMPTST' AND describe_ IN ('CM')
order by sl,pl,mape_i,mse,mae;

SELECT sl,pl,model,mse,mae,mape_i
INTO OUTFILE './168-ps10.0.csv'
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
FROM exp_results_t4
WHERE sl = 168
  AND describe_ REGEXP '^(CM|pw)(10\\.0)$'
order by pl,mse;

SHOW VARIABLES LIKE "secure_file_priv";

SELECT dataset,sl,pl,describe_,model, mse, mae, mape_i
FROM exp_results_t4
WHERE sl = 168 AND describe_ IN ('CM10.0','CM','Exp','pw10.0')
order by sl,pl,model,mse;

SELECT dataset,sl,pl,describe_,model, mse, mae, mape_i
FROM exp_results_t4
WHERE model = 'TimesNet' and sl = 168
order by pl;

---------------------------------------------------------------------------------------------

USE AMPTST_result;
-- 1. 建表
CREATE TABLE ps_loss_promotion_v3 (
    dataset      VARCHAR(20),
    sl           INT,
    pl           INT,
    model        VARCHAR(20),
    describe_    VARCHAR(64),          -- 原始 10.0/6.0/2.0 串
    mse_promotion  DOUBLE,
    mae_promotion  DOUBLE,
    mape_i_promotion DOUBLE,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (dataset, sl, pl, model, describe_)
);

UPDATE exp_results_v3
SET describe_ = 'Exp'
WHERE describe_ IN ('Expv3');

-- 2. 生成并插入提升百分比
INSERT INTO ps_loss_promotion_v3
(dataset, sl, pl, model, describe_,
 mse_promotion, mae_promotion, mape_i_promotion)
WITH base AS (
    SELECT *
    FROM exp_results_v3
    WHERE describe_ REGEXP '^(CM|Exp)$'          -- 基线：CM 或 Exp
), promo AS (
    SELECT *
    FROM exp_results_v3
    WHERE describe_ REGEXP '^(CM|pw)(2\\.0|6\\.0|10\\.0)$'  -- 带数字的实验
)
SELECT p.dataset,
       p.sl,
       p.pl,
       p.model,
       REGEXP_SUBSTR(p.describe_, '[0-9]+\\.[0-9]+') AS describe_, -- 提取 2.0/6.0/10.0
       -- 提升百分比 = (基线 - 实验) / 基线 * 100
       ROUND((b.mse   - p.mse)   / b.mse   * 100, 4) AS mse_promotion,
       ROUND((b.mae   - p.mae)   / b.mae   * 100, 4) AS mae_promotion,
       ROUND((b.mape_i - p.mape_i) / b.mape_i * 100, 4) AS mape_i_promotion
FROM promo p
JOIN base b
  ON p.dataset = b.dataset
 AND p.sl      = b.sl
 AND p.pl      = b.pl
 AND p.model   = b.model
 AND b.describe_ = 'Exp';   -- 以 CM 为统一基线（没有 CM 时换 Exp 即可）

SELECT sl,pl,describe_,model,mse_promotion,mae_promotion,mape_i_promotion
FROM ps_loss_promotion_v1
where sl =168 and describe_ = '10.0'
order by pl,describe_,mae_promotion;

SELECT
    model,
    REGEXP_SUBSTR(describe_, '[0-9]+\\.[0-9]+') AS val,  -- 2.0/6.0/10.0
    sl,
    COUNT(*)                                    AS cnt,
    ROUND(AVG(mse_promotion), 4)  AS avg_mse_promotion,
    ROUND(AVG(mae_promotion), 4)  AS avg_mae_promotion,
    ROUND(AVG(mape_i_promotion), 4) AS avg_mape_promotion
FROM ps_loss_promotion_v3
WHERE sl IN (96, 168)                      -- 只关心这两档序列长度
  AND describe_ REGEXP '(2\\.0|6\\.0|10\\.0)'  -- 只保留带数字的实验
GROUP BY model, val, sl
ORDER BY sl,val,avg_mse_promotion,avg_mae_promotion,avg_mape_promotion;

SELECT
    model,
    sl,
    COUNT(*)                                    AS cnt,
    ROUND(AVG(mse_promotion), 4)  AS avg_mse_promotion,
    ROUND(AVG(mae_promotion), 4)  AS avg_mae_promotion,
    ROUND(AVG(mape_i_promotion), 4) AS avg_mape_promotion
FROM ps_loss_promotion_v3
WHERE sl IN (168)          -- 只这两档序列长度
  AND describe_ = '10.0'       -- <<< 仅关注 10.0 的强度
GROUP BY model, sl
ORDER BY  sl,avg_mse_promotion,avg_mae_promotion,avg_mape_promotion,model;