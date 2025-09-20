#!/usr/bin/env python3
import os
import re
import pymysql
from dotenv import load_dotenv

load_dotenv()

DB_CFG = dict(
    host=os.getenv('DB_HOST'),
    port=int(os.getenv('DB_PORT')),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASS'),
    database=os.getenv('DB_NAME'),
    charset='utf8mb4'
)

BLOCK_SEP = re.compile(r'\n\s*\n')          # 空行分块
PARAM_RE  = re.compile(r'(?P<key>[a-z]+)(?P<val>\d+)', re.I)  # 提取 xx_nn
METRIC_RE = re.compile(r'(mse|mae|mape_i):([\d\.e\-]+)', re.I)

def parse_block(block: str):
    """返回 dict，key 与表字段对应；解析失败返回 None"""
    lines = [L.strip() for L in block.splitlines() if L.strip()]
    if len(lines) != 2:
        return None
    param_line, metric_line = lines

    # 1. 解析参数
    params = {}
    for m in PARAM_RE.finditer(param_line):
        key, val = m.group('key'), int(m.group('val'))
        params[key] = val
    # 特殊布尔值 distil
    params['dt'] = 'dtTrue' in param_line

    # 2. 解析指标
    metrics = {}
    for m in METRIC_RE.finditer(metric_line):
        metrics[m.group(1).lower()] = float(m.group(2))

    # 3. 合并
    row = {**params, **metrics}
    # 检查必要字段
    need_cols = {'h','sl','pl','mse','mae','mape_i'}
    if not need_cols.issubset(row):
        return None
    return row

def insert_many(rows):
    """批量写入，冲突则忽略"""
    if not rows:
        return
    cols = ['h','sl','pl','model','ft','ll','dm','nh','el','dl','df','fc','ebtime','dt','cm',
            'mse','mae','mape_i']
    # 缺省字段补 NULL
    for r in rows:
        for c in cols:
            r.setdefault(c, None)
    placeholders = ','.join(['%s']*len(cols))
    update_part  = ','.join([f'{c}=VALUES({c})' for c in cols if c not in {'h','sl','pl','model','ft','ll','dm','nh','el','dl','df','fc','ebtime','dt','cm'}])
    sql = f"""
    INSERT INTO exp_results_t1 ({','.join(cols)})
    VALUES ({placeholders})
    ON DUPLICATE KEY UPDATE {update_part}
    """
    args = [[row[c] for c in cols] for row in rows]
    with pymysql.connect(**DB_CFG) as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, args)
        conn.commit()

def main(file_path='./result_v1.txt'):
    with open(file_path, encoding='utf-8') as f:
        content = f.read()
    blocks = BLOCK_SEP.split(content)
    rows = [r for b in blocks if (r := parse_block(b))]
    insert_many(rows)
    print(f'loaded {len(rows)} records into MySQL.')

if __name__ == '__main__':
    main()
