import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

with open('timing_data/train_time_GP.csv') as f:
    lines = f.readlines()
    GP_train = np.array(lines, dtype = np.float32)


with open('timing_data/test_time_GP.csv') as f:
    lines = f.readlines()
    GP_test = np.array(lines, dtype = np.float32)


with open('timing_data/train_time_Ridge.csv') as f:
    lines = f.readlines()
    Ridge_train = np.array(lines, dtype = np.float32)


with open('timing_data/test_time_Ridge.csv') as f:
    lines = f.readlines()
    Ridge_test = np.array(lines, dtype = np.float32)


with open('timing_data/train_time_svm.csv') as f:
    lines = f.readlines()
    SVM_train = np.array(lines, dtype = np.float32)


with open('timing_data/test_time_svm.csv') as f:
    lines = f.readlines()
    SVM_test = np.array(lines, dtype = np.float32)


with open('timing_data/train_time_pca2.csv') as f:
    lines = f.readlines()
    pca2_train = np.array(lines, dtype = np.float32)


with open('timing_data/test_time_pca2.csv') as f:
    lines = f.readlines()
    pca2_test = np.array(lines, dtype = np.float32)


with open('timing_data/train_time_pca3.csv') as f:
    lines = f.readlines()
    pca3_train = np.array(lines, dtype = np.float32)


with open('timing_data/test_time_pca3.csv') as f:
    lines = f.readlines()
    pca3_test = np.array(lines, dtype = np.float32)

Ridge = Ridge_train+Ridge_test
GP = GP_train+GP_test
SVM = SVM_train+SVM_test
pca2 = pca2_train+pca2_test
pca3 = pca3_train+pca3_test


print(np.average(Ridge_train))
print(np.average(Ridge_test))

print(np.average(GP_train))
print(np.average(GP_test))

print(np.average(SVM_train))
print(np.average(SVM_test))

print(np.average(pca2_train))
print(np.average(pca2_test))

print(np.average(pca3_train))
print(np.average(pca3_test))


tstat, pval = ttest_ind(Ridge,GP)
print("T-statistic, p-value for Ridge and GP: (%f %g) " % (tstat,pval))

tstat, pval = ttest_ind(Ridge,SVM)
print("T-statistic, p-value for Ridge and SVM: (%f %g) " % (tstat,pval))

tstat, pval = ttest_ind(Ridge,pca2)
print("T-statistic, p-value for Ridge and 2PCR: (%f %g) " % (tstat,pval))

tstat, pval = ttest_ind(Ridge,pca3)
print("T-statistic, p-value for Ridge and 3PCR: (%f %g) " % (tstat,pval))

tstat, pval = ttest_ind(GP,SVM)
print("T-statistic, p-value for GP and SVM: (%f %g) " % (tstat,pval))

tstat, pval = ttest_ind(GP,pca2)
print("T-statistic, p-value for GP and pca2: (%f %g) " % (tstat,pval))

tstat, pval = ttest_ind(GP,pca3)
print("T-statistic, p-value for GP and pca3: (%f %g) " % (tstat,pval))

tstat, pval = ttest_ind(SVM,pca2)
print("T-statistic, p-value for SVM and pca2: (%f %g) " % (tstat,pval))

tstat, pval = ttest_ind(SVM,pca3)
print("T-statistic, p-value for SVM and pca3: (%f %g) " % (tstat,pval))

tstat, pval = ttest_ind(pca2,pca3)
print("T-statistic, p-value for pca2 and pca3: (%f %g) " % (tstat,pval))

plt.hist(GP_train)
plt.xlabel('Training Time (sec)')
plt.ylabel('Count')
plt.title('Histogram of GPR Training')
plt.savefig('graphs/GP_train.png')
plt.show()


plt.hist(GP_test)
plt.xlabel('Testing Time (sec)')
plt.ylabel('Count')
plt.title('Histogram of GPR Testing')
plt.savefig('graphs/GP_test.png')
plt.show()


plt.hist(Ridge_train)
plt.xlabel('Training Time (sec)')
plt.ylabel('Count')
plt.title('Histogram of Ridge Regression Training')
plt.savefig('graphs/Ridge_train.png')
plt.show()


plt.hist(Ridge_test)
plt.xlabel('Testing Time (sec)')
plt.ylabel('Count')
plt.title('Histogram of Ridge Regression Testing')
plt.savefig('graphs/Ridge_test.png')
plt.show()


plt.hist(SVM_train)
plt.xlabel('Training Time (sec)')
plt.ylabel('Count')
plt.title('Histogram of SVM Regression Training')
plt.savefig('graphs/SVM_train.png')
plt.show()


plt.hist(SVM_test)
plt.xlabel('Testing Time (sec)')
plt.ylabel('Count')
plt.title('Histogram of SVM Regression Testing')
plt.savefig('graphs/svm_test.png')
plt.show()


plt.hist(pca2_train)
plt.xlabel('Training Time (sec)')
plt.ylabel('Count')
plt.title('Histogram of Principal 2-Component Regression Training')
plt.savefig('graphs/pca2_train.png')
plt.show()


plt.hist(pca2_test)
plt.xlabel('Testing Time (sec)')
plt.ylabel('Count')
plt.title('Histogram of Principal 2-Component Regression Testing')
plt.savefig('graphs/pca2_test.png')
plt.show()


plt.hist(pca3_train)
plt.xlabel('Training Time (sec)')
plt.ylabel('Count')
plt.title('Histogram of Principal 3-Component Regression Training')
plt.savefig('graphs/pca3_train.png')
plt.show()


plt.hist(pca3_test)
plt.xlabel('Testing Time (sec)')
plt.ylabel('Count')
plt.title('Histogram of Principal 3-Component Regression Testing')
plt.savefig('graphs/pca3_test.png')
plt.show()
