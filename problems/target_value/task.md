# Target Value

Find a Python program that defines a variable `result` whose value is as close
to 42.0 as possible. The reward is `1 / (1 + |result - 42|)`, so the optimum
is 1.0 at `result == 42`, and any invalid program gets 0.0.
