# Steering Vector Calculation with np.einsum

## Overview

`np.einsum`はアインシュタインの縮約記法に基づくNumPyの関数。多次元配列（テンソル）間で複雑な操作をシンプルかつ効率的に行える。

## Function Signature

```python
numpy.einsum(subscripts, *operands, out=None, dtype=None, order='K', casting='safe', optimize=False)
```

- `subscripts`: 操作指示用の文字列
- `*operands`: 操作対象のテンソル

## Subscripts

### Flexibility in Labels

添字に使う文字（`k`, `i`, `s`, `m`など）は基本的に任意。ただし、同じ文字は次元が一致している必要があり、その次元で縮約（和）が行われる。

### Conventional Labels

添字に使われる文字には一般的な慣習がある。例えば、`k`は周波数、`t`は時間、`i`や`j`は一般的なインデックス、`m`はマイクロフォンやモデル、`s`はソースやシグナルなど。

## Example

```python
np.einsum("k,ism,ism->ksm", freqs, norm_source_locations, mic_alignments)
```

- `k`: `freqs`（1次元）
- `ism`: `norm_source_locations`（3次元）
- `ism`: `mic_alignments`（3次元）

## Interpretation

指定`"k,ism,ism->ksm"`は、`k`, `i`, `s`, `m`各次元で、`ism`次元で和を取り、新しいテンソルを`ksm`形で生成する。

具体的には、次のような計算が行われる。

```
Output[k, s, m] = Σ_i (freqs[k] * norm_source_locations[i, s] * mic_alignments[i, m])
```

この計算により、`i`次元は縮約されて消える。

## Performance

`timeit`で計測した結果、`np.einsum`の方が明らかに高速。

- `np.einsum`: 0.486秒
- 縮約なし: 10.49秒

## Conclusion

`np.einsum`はテンソル計算を効率的に行える強力なツール。特に縮約操作が必要な場合に有用。添字は基本的に任意だが、一般的な慣習に基づく命名も存在する。そして、計算速度も非常に高い。
