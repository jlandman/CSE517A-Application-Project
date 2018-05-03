import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

with open('error_data/train_error_GP.csv') as f:
    lines = f.readlines()
    GP_train = np.array(lines, dtype = np.float32)


with open('error_data/test_error_GP.csv') as f:
    lines = f.readlines()
    GP_test = np.array(lines, dtype = np.float32)


with open('error_data/train_error_Ridge.csv') as f:
    lines = f.readlines()
    Ridge_train = np.array(lines, dtype = np.float32)


with open('error_data/test_error_Ridge.csv') as f:
    lines = f.readlines()
    Ridge_test = np.array(lines, dtype = np.float32)


with open('error_data/train_error_svm.csv') as f:
    lines = f.readlines()
    SVM_train = np.array(lines, dtype = np.float32)


with open('error_data/test_error_svm.csv') as f:
    lines = f.readlines()
    SVM_test = np.array(lines, dtype = np.float32)

with open('error_data/train_error_pca2.csv') as f:
    lines = f.readlines()
    pca2_train = np.array(lines, dtype = np.float32)


with open('error_data/test_error_pca2.csv') as f:
    lines = f.readlines()
    pca2_test = np.array(lines, dtype = np.float32)


with open('error_data/train_error_pca3.csv') as f:
    lines = f.readlines()
    pca3_train = np.array(lines, dtype = np.float32)


with open('error_data/test_error_pca3.csv') as f:
    lines = f.readlines()
    pca3_test = np.array(lines, dtype = np.float32)

GP_train_av = np.average(GP_train)
GP_test_av = np.average(GP_test)
Ridge_train_av = np.average(Ridge_train)
Ridge_test_av = np.average(Ridge_test)
SVM_train_av = np.average(SVM_train)
SVM_test_av = np.average(SVM_test)
pca2_train_av = np.average(pca2_train)
pca2_test_av = np.average(pca2_test)
pca3_train_av = np.average(pca3_train)
pca3_test_av = np.average(pca3_test)




tstat, pval = ttest_ind(Ridge_test,GP_test) 
print("T-statistic, p-value for Ridge and GP: (%f %g) " % (tstat,pval)) 
 
tstat, pval = ttest_ind(Ridge_test,SVM_test) 
print("T-statistic, p-value for Ridge and SVM: (%f %g) " % (tstat,pval)) 
 
tstat, pval = ttest_ind(Ridge_test,pca2_test) 
print("T-statistic, p-value for Ridge and 2PCR: (%f %g) " % (tstat,pval)) 
 
tstat, pval = ttest_ind(Ridge_test,pca3_test) 
print("T-statistic, p-value for Ridge and 3PCR: (%f %g) " % (tstat,pval)) 
 
tstat, pval = ttest_ind(GP_test,SVM_test) 
print("T-statistic, p-value for GP and SVM: (%f %g) " % (tstat,pval)) 
 
tstat, pval = ttest_ind(GP_test,pca2_test) 
print("T-statistic, p-value for GP and pca2: (%f %g) " % (tstat,pval)) 
 
tstat, pval = ttest_ind(GP_test,pca3_test) 
print("T-statistic, p-value for GP and pca3: (%f %g) " % (tstat,pval)) 
 
tstat, pval = ttest_ind(SVM_test,pca2_test) 
print("T-statistic, p-value for SVM and pca2: (%f %g) " % (tstat,pval)) 
 
tstat, pval = ttest_ind(SVM_test,pca3_test) 
print("T-statistic, p-value for SVM and pca3: (%f %g) " % (tstat,pval)) 
 
tstat, pval = ttest_ind(pca2_test,pca3_test) 
print("T-statistic, p-value for pca2 and pca3: (%f %g) " % (tstat,pval)) 





plt.scatter(GP_train, GP_test, label="samples")
plt.scatter([GP_train_av], [GP_test_av], label="average")
plt.xlabel("Training MAE")
plt.ylabel("Testing MAE")
plt.title('GP Regression Errors')
plt.legend()
plt.savefig('graphs/GP.png')
plt.show()


plt.scatter(Ridge_train, Ridge_test, label = "samples")
plt.scatter([Ridge_train_av], [Ridge_test_av], label="average")
plt.xlabel("Training MAE")
plt.ylabel("Testing MAE")
plt.title('Ridge Regression Errors')
plt.legend()
plt.savefig('graphs/Ridge.png')
plt.show()


plt.scatter(SVM_train, SVM_test, label = "samples")
plt.scatter([SVM_train_av], [SVM_test_av], label="average")
plt.xlabel("Training MAE")
plt.ylabel("Testing MAE")
plt.title('SVM Regression Errors')
plt.legend()
plt.savefig('graphs/SVM.png')
plt.show()



plt.scatter(pca2_train, pca2_test, label = "samples")
plt.scatter([pca2_train_av], [pca2_test_av], label="average")
plt.xlabel("Training MAE")
plt.ylabel("Testing MAE")
plt.title('Principal 2-Component Regression Errors')
plt.legend()
plt.savefig('graphs/pca2.png')
plt.show()



plt.scatter(pca3_train, pca3_test, label = "samples")
plt.scatter([pca3_train_av], [pca3_test_av], label="average")
plt.xlabel("Training MAE")
plt.ylabel("Testing MAE")
plt.title('Principal 3-Component Regression Errors')
plt.legend()
plt.savefig('graphs/pca3.png')
plt.show()




plt.scatter(Ridge_train, Ridge_test, label = "Ridge samples")
plt.scatter(GP_train, GP_test, label="GP samples")
plt.scatter(SVM_train, SVM_test, label="SVM samples")
plt.scatter(pca2_train, pca2_test, label = "2PCR samples")
plt.scatter(pca3_train, pca3_test, label = "3PCR samples")
plt.xlabel("Training MAE")
plt.ylabel("Testing MAE")
plt.title('Comparative Errors')
plt.legend(loc='upper left')
plt.savefig('graphs/Comparison.png')
plt.show()

