package com.mass.scalann.tensor

trait TensorMath[T] {

  def addmv(alpha: T, mat: Tensor[T], vec: Tensor[T]): Tensor[T]

  def addmv(beta: T, alpha: T, mat: Tensor[T], vec: Tensor[T]): Tensor[T]

  def addmm(v1: T, M: Tensor[T], v2: T, mat1: Tensor[T], mat2: Tensor[T]): Tensor[T]

  def addmm(v1: T, v2: T, mat1: Tensor[T], mat2: Tensor[T]): Tensor[T]

  def addmm(v: T, mat1: Tensor[T], mat2: Tensor[T]): Tensor[T]

  def bmm(beta: T, alpha: T, mat1: Tensor[T], mat2: Tensor[T]): Tensor[T]

  def addr(v: T, vec1: Tensor[T], vec2: Tensor[T]): Tensor[T]

  def mul(value: T): Tensor[T]

  def mul(x: Tensor[T], value: T): Tensor[T]

  def div(value: T): Tensor[T]

  // Element-wise multiply
  def cmul(y: Tensor[T]): Tensor[T]

  def cmul(x: Tensor[T], y: Tensor[T]): Tensor[T]

  // Element-wise divide
  def cdiv(y: Tensor[T]): Tensor[T]

  def cdiv(x: Tensor[T], y: Tensor[T]): Tensor[T]

  // res = res + value * (tensor1 / tensor2)
  def addcdiv(value: T, tensor1: Tensor[T], tensor2: Tensor[T]): Tensor[T]

  def add(value: T, y: Tensor[T]): Tensor[T]

  def add(value: T): Tensor[T]

  def add(y: Tensor[T]): Tensor[T]

  def sub(y : Tensor[T]) : Tensor[T]

  def sub(value : T) : Tensor[T]

  def log(): Tensor[T]

  def pow(y: Tensor[T], n : T): Tensor[T]

  def pow(n: T): Tensor[T]

  def exp(): Tensor[T]

  def exp(y: Tensor[T]): Tensor[T]

  def sqrt(y: Tensor[T]): Tensor[T]

  def dot(y: Tensor[T]): T

  def sum(): T

  def sum(dim: Int): Tensor[T]

  def max(): T

}
