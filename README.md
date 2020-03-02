# Benchmarkfcns
This is a reimplementation of [benchmarkfcns](https://github.com/mazhar-ansari-ardeh/BenchmarkFcns) for use with [CasADi](https://web.casadi.org/). The Python module is intended to be used to generate the required functions as either `MX` or `SX` CasADi functions. Example usage:

```
import casadi as cs
import benchmarkfcns2casadi as bm

rosenbrock_func, input_domains, minima = generate_rosenbrock(n=2, a=1, b=100, data_type=cs.SX)
print(rosenbrock_func)
```
