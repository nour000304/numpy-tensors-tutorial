# 🚀 Comprehensive NumPy & Tensors Guide

Welcome to the **NumPy & Tensors Tutorial**! This guide covers everything from basic array initialization and dimensional intuition (scalars to 3D+ tensors) to advanced concepts like memory management (`copy` vs. `view`) and vector broadcasting.

---

## 📚 Table of Contents
1. [Data Representation & Shapes](#1-data-representation--shapes)
2. [Array Properties & Accessing Elements](#2-array-properties--accessing-elements)
3. [Vector Operations & Broadcasting](#3-vector-operations--broadcasting)
4. [Memory Management: Copy vs. View](#4-memory-management-copy-vs-view)
5. [Statistics & Filtering](#5-statistics--filtering)

---

## 1. Data Representation & Shapes

Understanding dimensional shapes ($\text{NDIM}$) is critical for machine learning and numerical analysis[cite: 1].

| Type | Shape (`.shape`) | Example Concept | Code Syntax |
| :--- | :--- | :--- | :--- |
| **Scalar** | `()` | 0D: Single value | `np.array(5)`[cite: 1] |
| **Vector** | `(n,)` | 1D: Line | `np.array([3, 5, 6])`[cite: 1] |
| **Matrix** | `(m, n)` | 2D: Table / Sheet | `np.array([[2, 5], [4, 5]])`[cite: 1] |
| **Tensor** | `(x, y, z)` | 3D+: Stack of Sheets / Video | `np.array([[[1, 2]], [[5, 6]]])`[cite: 1] |

### Visualizing Tensor Dimensions
* 🟩 **1D Tensor**: Line $[3, 5, 6]$[cite: 1]
* 🟦 **2D Tensor**: Sheet (Rows $\times$ Columns)[cite: 1]
* 🟥 **3D Tensor**: Stack of sheets[cite: 1]
* 🟪 **4D Tensor**: Stack of stacks (e.g., video frames)[cite: 1]

---

## 2. Array Properties & Accessing Elements

### Quick Property Diagnostics
```python
import numpy as np

m = np.array([[2, 5, 6], [4, 5, 6]])[cite: 1]

print(m.ndim)   # Number of dimensions[cite: 1]
print(m.shape)  # Structural layout (rows, cols)[cite: 1]
print(m.size)   # Total element count[cite: 1]
print(m.dtype)  # Data type of elements[cite: 1]
print(m.nbytes) # Total memory consumed (bytes)[cite: 1]
```

### Slicing & Indexing
```python
a = np.array([
    [1, 2, 3, 4, 5, 6, 7],
    [8, 9, 10, 11, 12, 13, 14]
])[cite: 1]

# Specific Element -> [row, col]
element = a[1, 3] # Returns 11[cite: 1]

# Specific Column
col = a[:, 2] # Returns array([3, 10])[cite: 1]

# Stepped Slicing [start:stop:step]
slice_sample = a[0, 1:6:2] # Returns array([2, 4, 6])[cite: 1]
```

---

## 3. Vector Operations & Broadcasting

### Element-wise vs. Matrix Multiplication
```python
a = np.array([1, 2, 3])[cite: 1]
b = np.array([4, 5, 6])[cite: 1]

# Vector Dot Product
dot_product = np.dot(a, b) # or (a @ b)[cite: 1]

# Element-wise Multiplication
elem_mult = a * b # array([ 4, 10, 18])[cite: 1]
```

### Broadcasting Intuition
Broadcasting allows NumPy to perform operations on arrays of different shapes without explicit loops[cite: 1].

```text
Array A (2, 3):             Array B (3,):            Broadcasted B:
[[ 1,  2,  3],      +       [10, 20, 30]     -->     [[10, 20, 30],
 [ 4,  5,  6]]                                        [10, 20, 30]][cite: 1]
```

---

## 4. Memory Management: Copy vs. View

| Method | Syntax | Memory Allocation | Safety |
| :--- | :--- | :--- | :--- |
| **View (Ravel)** | `r = a.ravel()` or `c = a.view()`[cite: 1] | **Shared memory** (Modifying `r` changes `a`)[cite: 1] | ⚡ Fast / High efficiency[cite: 1] |
| **Copy (Flatten)**| `f = a.flatten()` or `b = a.copy()`[cite: 1] | **New memory allocated** (Independent object)[cite: 1] | 🛡️ Safe / Independent[cite: 1] |

---

## 💡 Quick Summary Cheat Sheet

* **`reshape()`**: Changes layout without altering underlying data[cite: 1].
* **`flatten()` vs `ravel()`**: Use `flatten()` for safe copies; use `ravel()` for high-speed views[cite: 1].
* **Broadcasting**: Eliminates explicit `for` loops in matrix arithmetic[cite: 1].
* **Dot Product**: Use `a @ b` or `np.dot(a, b)`[cite: 1].
