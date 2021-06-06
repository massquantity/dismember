package com.mass.tdm.utils

object Utils {
  // import scala.concurrent.duration._
  // import java.util.concurrent.TimeUnit
  // (end - start).nanos.toSeconds
  // TimeUnit.NANOSECONDS.toSeconds(end - start)

  def time[T](block: => T, info: String): T = {
    val t0 = System.nanoTime()
    val result = block
    val t1 = System.nanoTime()
    println(f"$info time: ${(t1 - t0) / 1e9d}%.4fs")
    result
  }

  // Note this algorithm is inplace
  def argSort[@specialized(Int, Long, Float, Double) T](elems: Array[T], indices: Array[Int])(
      implicit order: T => Ordered[T]): Unit = {

    sort(0, elems.length)

    def swap(a: Int, b: Int): Unit = {
      val tmp = elems(a)
      elems(a) = elems(b)
      elems(b) = tmp
      val tmp2 = indices(a)
      indices(a) = indices(b)
      indices(b) = tmp2
    }

    def vecswap(_a: Int, _b: Int, n: Int): Unit = {
      var a = _a
      var b = _b
      var i = 0
      while (i < n) {
        swap(a, b)
        i += 1
        a += 1
        b += 1
      }
    }

    def midIdx(a: Int, b: Int, c: Int): Int = {
      if (elems(a) < elems(b)) {
        if (elems(b) < elems(c)) b else if (elems(a) < elems(c)) c else a
      } else {
        if (elems(b) > elems(c)) b else if (elems(a) > elems(c)) c else a
      }
    }

    def sort(off: Int, length: Int): Unit = {
      // Insertion sort on smallest arrays
      if (length < 7) {
        var i = off
        while (i < length + off) {
          var j = i
          while (j > off && elems(j - 1) > elems(j)) {
            swap(j, j - 1)
            j -= 1
          }
          i += 1
        }
      } else {
        // Choose a partition element, v
        var m = off + (length >> 1) // Small arrays, middle element
        if (length > 7) {
          var l = off
          var n = off + length - 1
          if (length > 40) { // Big arrays, pseudomedian of 9
            val s = length / 8
            l = midIdx(l, l + s, l + 2 * s)
            m = midIdx(m - s, m, m + s)
            n = midIdx(n - 2 * s, n - s, n)
          }
          m = midIdx(l, m, n) // Mid-size, med of 3
        }
        val v = elems(m)

        // Establish Invariant: v* (<v)* (>v)* v*
        var a = off
        var b = a
        var c = off + length - 1
        var d = c
        var done = false
        while (!done) {
          while (b <= c && elems(b) <= v) {
            if (elems(b) == v) {
              swap(a, b)
              a += 1
            }
            b += 1
          }
          while (c >= b && elems(c) >= v) {
            if (elems(c) == v) {
              swap(c, d)
              d -= 1
            }
            c -= 1
          }
          if (b > c) {
            done = true
          } else {
            swap(b, c)
            c -= 1
            b += 1
          }
        }

        // Swap partition elements back to middle
        val n = off + length
        var s = math.min(a - off, b - a)
        vecswap(off, b - s, s)
        s = math.min(d - c, n - d - 1)
        vecswap(b, n - s, s)

        // Recursively sort non-partition-elements
        s = b - a
        if (s > 1)
          sort(off, s)
        s = d - c
        if (s > 1)
          sort(n - s, s)
      }
    }
  }

  // Note this algorithm is inplace
  def argPartition[@specialized(Int, Long, Float, Double) T](
      elems: Array[T],
      position: Int,
      indices: Array[Int])(implicit order: T => Ordered[T]): Unit = {

    selectSort(elems, position)

    def selectSort(elems: Array[T], position: Int): Unit = {
      var left = 0
      var right = elems.length - 1
      require(position >= left && position <= right)

      while (left < right) {
        val pvt = medIdx(left, right, ((left.toLong + right) / 2).toInt)
        val (lt, gt) = partition3Ways(elems, left, right, pvt)
        if (lt <= position && position <= gt) {
          left = right
        } else if (position < lt) {
          right = lt - 1
        } else if (position > gt) {
          left = gt + 1
        }
      }
    }

    def medIdx(p1: Int, p2: Int, p3: Int) = {
      if (elems(p1) < elems(p2)) {
        if (elems(p2) < elems(p3)) p2 else if (elems(p1) < elems(p3)) p3 else p1
      } else {
        if (elems(p2) > elems(p3)) p2 else if (elems(p1) > elems(p3)) p3 else p1
      }
    }

    def partition3Ways(elems: Array[T], left: Int, right: Int, pivot: Int): (Int, Int) = {
      val pivotVal = elems(pivot)
      swap(pivot, left)
      var i = left
      var lt = left
      var gt = right

      while (i <= gt) {
        if (elems(i) < pivotVal) {
          swap(lt, i)
          lt += 1
          i += 1
        } else if (elems(i) > pivotVal) {
          swap(gt, i)
          gt -= 1
        } else if (elems(i) == pivotVal) {
          i += 1
        } else {
          assert(elems(i) != elems(i))
          throw new IllegalArgumentException("Nan element detected")
        }
      }

      (lt, gt)
    }

    def swap(a: Int, b: Int): Unit = {
      val tmp = elems(a)
      elems(a) = elems(b)
      elems(b) = tmp
      val tmp2 = indices(a)
      indices(a) = indices(b)
      indices(b) = tmp2
    }
  }
}
