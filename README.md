# AutoVQC

変分量子回路の高速自動最適化ツール

## 例

```python
from autovqc import CircuitOptimizer

circ_opt = CircuitOptimizer(
    # 入力① ハミルトニアン(辞書型)
    hamiltonian={'XIXI': 0.1, 'XIZZ': 0.3, ...},
    # 入力② 隣接する量子ビットのペアのリスト
    connections=[(0, 1), (1, 3), (2, 3), ...],
    # 入力③ 量子回路の深さ
    n_depth=5
)
for i in range(10):
    circ_opt.update() # 変分量子回路の構成を更新
    
res = circ_opt.get_result() # 結果を出力(辞書型)

# 出力① ハミルトニアンの基底エネルギー
res['loss'] #=> -0.869...
# 出力② 最適化された変分量子回路の構成
res['targets_list']  #=> [(0, 1), (2, 3), ...],
# 出力③ 最適化された変分量子回路パラメータ
res['params']  #=> [6.12, 3.22, ...]
```