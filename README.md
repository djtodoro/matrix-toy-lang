# matrix-toy-lang
A Toy Lang to learn MLIR.

## Enviroment

```bash
python -m venv venv
source venv/bin/activate
```

## Install

```bash
pip install -e .
```

## Example

Without optimization:

```bash
(venv) $ matrixc --no-optimize examples/main_input.mx 
Compiling examples/main_input.mx...
Parsed 2 functions with 10 operations
Generated initial IR
Skipping optimizations

Generated IR:
builtin.module {
  func.func @matrix_computation(%0 : !matrix.type<#builtin.int<4>, #builtin.int<4>, f32>, %1 : !matrix.type<#builtin.int<4>, #builtin.int<4>, f32>) -> !matrix.type<#builtin.int<4>, #builtin.int<4>, f32> {
    %2 = "matrix.transpose"(%0) : (!matrix.type<#builtin.int<4>, #builtin.int<4>, f32>) -> !matrix.type<#builtin.int<4>, #builtin.int<4>, f32>
    %3 = "matrix.transpose"(%2) : (!matrix.type<#builtin.int<4>, #builtin.int<4>, f32>) -> !matrix.type<#builtin.int<4>, #builtin.int<4>, f32>
    %4 = "matrix.transpose"(%1) : (!matrix.type<#builtin.int<4>, #builtin.int<4>, f32>) -> !matrix.type<#builtin.int<4>, #builtin.int<4>, f32>
    %5 = "matrix.transpose"(%4) : (!matrix.type<#builtin.int<4>, #builtin.int<4>, f32>) -> !matrix.type<#builtin.int<4>, #builtin.int<4>, f32>
    %6 = "matrix.add"(%3, %5) : (!matrix.type<#builtin.int<4>, #builtin.int<4>, f32>, !matrix.type<#builtin.int<4>, #builtin.int<4>, f32>) -> !matrix.type<#builtin.int<4>, #builtin.int<4>, f32>
    %7 = "matrix.transpose"(%6) : (!matrix.type<#builtin.int<4>, #builtin.int<4>, f32>) -> !matrix.type<#builtin.int<4>, #builtin.int<4>, f32>
    %8 = "matrix.transpose"(%7) : (!matrix.type<#builtin.int<4>, #builtin.int<4>, f32>) -> !matrix.type<#builtin.int<4>, #builtin.int<4>, f32>
    func.return %8 : !matrix.type<#builtin.int<4>, #builtin.int<4>, f32>
  }
  func.func @main() {
    %0 = "matrix.alloc"() {shape = [#builtin.int<3>, #builtin.int<3>]} : () -> !matrix.type<#builtin.int<3>, #builtin.int<3>, f32>
    %1 = "matrix.alloc"() {shape = [#builtin.int<3>, #builtin.int<3>]} : () -> !matrix.type<#builtin.int<3>, #builtin.int<3>, f32>
    %2 = "matrix.add"(%0, %1) : (!matrix.type<#builtin.int<3>, #builtin.int<3>, f32>, !matrix.type<#builtin.int<3>, #builtin.int<3>, f32>) -> !matrix.type<#builtin.int<3>, #builtin.int<3>, f32>
    func.return
  }
}
Compilation successful!
```

With optimization:

```bash
(venv) $ matrixc examples/main_input.mx 
Compiling examples/main_input.mx...
Parsed 2 functions with 10 operations
Generated initial IR
Running optimization passes...

Generated IR:
builtin.module {
  func.func @matrix_computation(%0 : !matrix.type<#builtin.int<4>, #builtin.int<4>, f32>, %1 : !matrix.type<#builtin.int<4>, #builtin.int<4>, f32>) -> !matrix.type<#builtin.int<4>, #builtin.int<4>, f32> {
    %2 = "matrix.transpose"(%0) : (!matrix.type<#builtin.int<4>, #builtin.int<4>, f32>) -> !matrix.type<#builtin.int<4>, #builtin.int<4>, f32>
    %3 = "matrix.transpose"(%1) : (!matrix.type<#builtin.int<4>, #builtin.int<4>, f32>) -> !matrix.type<#builtin.int<4>, #builtin.int<4>, f32>
    %4 = "matrix.add"(%0, %1) : (!matrix.type<#builtin.int<4>, #builtin.int<4>, f32>, !matrix.type<#builtin.int<4>, #builtin.int<4>, f32>) -> !matrix.type<#builtin.int<4>, #builtin.int<4>, f32>
    %5 = "matrix.transpose"(%4) : (!matrix.type<#builtin.int<4>, #builtin.int<4>, f32>) -> !matrix.type<#builtin.int<4>, #builtin.int<4>, f32>
    func.return %4 : !matrix.type<#builtin.int<4>, #builtin.int<4>, f32>
  }
  func.func @main() {
    %0 = "matrix.alloc"() {shape = [#builtin.int<3>, #builtin.int<3>]} : () -> !matrix.type<#builtin.int<3>, #builtin.int<3>, f32>
    %1 = "matrix.alloc"() {shape = [#builtin.int<3>, #builtin.int<3>]} : () -> !matrix.type<#builtin.int<3>, #builtin.int<3>, f32>
    %2 = "matrix.add"(%0, %1) : (!matrix.type<#builtin.int<3>, #builtin.int<3>, f32>, !matrix.type<#builtin.int<3>, #builtin.int<3>, f32>) -> !matrix.type<#builtin.int<3>, #builtin.int<3>, f32>
    func.return
  }
}
Compilation successful!
```
