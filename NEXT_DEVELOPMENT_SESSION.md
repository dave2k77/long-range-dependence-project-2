# Next Development Session - Action Plan

## 🎯 **Session Goal**
Fix the most critical test failures and get the overall test pass rate above 80% (from current 67.8%).

## 📊 **Current Status Summary**
- **Total Tests**: 180
- **Passing**: 122 (67.8%)
- **Failing**: 58 (32.2%)
- **Wavelet Estimators**: ✅ 31/31 (100% passing)
- **Other Estimators**: ⚠️ 91/149 (61.1% passing)

## 🚨 **Critical Issues to Fix (Priority 1)**

### **1. High-Performance Estimators - JAX Integration Issues**
- **Problem**: `TracerBoolConversionError` in JAX functions
- **Files**: `src/estimators/high_performance.py`
- **Tests Affected**: 15+ test failures
- **Root Cause**: Dynamic control flow in JAX-compiled functions
- **Solution**: Refactor to avoid conditional statements in JAX functions

### **2. Missing Attributes in Estimators**
- **Problem**: Tests expect attributes that don't exist
- **Examples**: 
  - `min_freq` in spectral estimators
  - `num_freq` in periodogram estimators
  - `n_bootstrap` in various estimators
- **Solution**: Add missing attributes with appropriate default values

### **3. Data Validation Logic**
- **Problem**: Missing validation for constant data
- **Tests Affected**: Multiple edge case tests
- **Solution**: Add constant data detection in `_validate_data()` methods

## 🔧 **Specific Fixes Needed (Priority 2)**

### **High-Performance DFA Estimator**
```python
# Fix in src/estimators/high_performance.py
def _validate_data(self):
    if self.data is None:
        raise ValueError("No data provided")
    # ... rest of validation
```

### **High-Performance MFDFA Estimator**
```python
# Fix JAX function in src/estimators/high_performance.py
# Replace conditional logic with JAX-compatible operations
def _calculate_fluctuation_jax(data, scale, q, polynomial_order):
    # Use jax.lax.cond instead of if statements
    # Or restructure to avoid dynamic control flow
```

### **Spectral Estimators**
```python
# Add missing attributes in src/estimators/spectral.py
class PeriodogramEstimator(BaseEstimator):
    def __init__(self, **kwargs):
        # ... existing code ...
        self.min_freq = kwargs.get('min_freq', 0.01)
        self.num_freq = kwargs.get('num_freq', 256)
```

### **Temporal Estimators**
```python
# Add constant data validation in src/estimators/temporal.py
def _validate_data(self):
    # ... existing validation ...
    if np.std(self.data) == 0:
        raise ValueError("Data is constant, cannot estimate LRD")
```

## 📋 **Step-by-Step Action Plan**

### **Step 1: Fix High-Performance Estimators (30 minutes)**
1. **Identify JAX Issues**: Look at the `TracerBoolConversionError` in `high_performance.py`
2. **Refactor Functions**: Replace conditional logic with JAX-compatible operations
3. **Test Fixes**: Run high-performance estimator tests to verify

### **Step 2: Add Missing Attributes (20 minutes)**
1. **Spectral Estimators**: Add `min_freq`, `num_freq` attributes
2. **Temporal Estimators**: Add `n_bootstrap` attributes where needed
3. **Verify**: Check that tests can access expected attributes

### **Step 3: Fix Data Validation (20 minutes)**
1. **Constant Data Detection**: Add validation for constant time series
2. **Edge Case Handling**: Improve validation for short data, NaN/inf values
3. **Test Validation**: Ensure edge case tests pass

### **Step 4: Test and Verify (15 minutes)**
1. **Run Focused Tests**: Test the specific estimators we fixed
2. **Check Pass Rate**: Verify improvement in test success rate
3. **Document Issues**: Note any remaining problems for next session

## 🎯 **Success Criteria for This Session**

### **Minimum Success** ✅
- Fix JAX integration issues in high-performance estimators
- Add missing attributes to all estimators
- Get test pass rate above 75%

### **Target Success** 🎯
- Fix all critical test failures
- Get test pass rate above 80%
- Have clear plan for remaining issues

### **Stretch Goal** 🚀
- Fix all estimator-related test failures
- Get test pass rate above 85%
- Ready for performance optimization phase

## 🔍 **Files to Focus On**

### **Primary Files** (Must Fix)
1. `src/estimators/high_performance.py` - JAX integration issues
2. `src/estimators/spectral.py` - Missing attributes
3. `src/estimators/temporal.py` - Validation logic

### **Secondary Files** (Nice to Fix)
1. `src/estimators/base.py` - Ensure consistent interfaces
2. `tests/` - Update test expectations if needed

### **Reference Files** (Don't Change)
1. `src/estimators/wavelet.py` - Working perfectly ✅
2. `src/estimators/base.py` - Core framework ✅

## ⚠️ **Potential Challenges**

### **JAX Complexity**
- **Challenge**: JAX compilation constraints are complex
- **Mitigation**: Start with simple fixes, use JAX documentation
- **Fallback**: Consider disabling JAX for problematic functions temporarily

### **Test Dependencies**
- **Challenge**: Fixing one issue might reveal others
- **Mitigation**: Fix systematically, test incrementally
- **Approach**: Fix one estimator at a time

### **Interface Consistency**
- **Challenge**: Different estimators have different interfaces
- **Mitigation**: Standardize gradually, maintain backward compatibility
- **Goal**: Consistent interface without breaking existing functionality

## 📚 **Resources for This Session**

### **JAX Documentation**
- [JAX Control Flow](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#control-flow)
- [JAX Best Practices](https://jax.readthedocs.io/en/latest/jax-101/01-jax-basics.html)

### **Research Papers**
- **Paper 1** (Wavelet Whittle) - For optimization ideas
- **Paper 5** (Hurst Algorithms) - For validation approaches

### **Current Code**
- Working wavelet estimators as reference implementation
- Base estimator framework for interface consistency

## 🚀 **Next Session Preview**

### **If This Session is Successful**
- Move to performance optimization phase
- Implement robust estimation methods
- Add trend-robust LRD estimation

### **If This Session Needs More Time**
- Continue fixing test failures
- Focus on one estimator family at a time
- Ensure solid foundation before optimization

---

**Session Duration**: 1.5 hours  
**Primary Goal**: Fix critical test failures  
**Success Metric**: Test pass rate > 80%  
**Next Milestone**: All core estimators working reliably

