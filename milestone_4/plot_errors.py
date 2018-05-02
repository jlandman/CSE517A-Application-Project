import numpy as np
import matplotlib.pyplot as plt

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
pca2_train_av = np.average(pca2_train)
pca2_test_av = np.average(pca2_test)
pca3_train_av = np.average(pca3_train)
pca3_test_av = np.average(pca3_test)

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
plt.scatter(pca2_train, pca2_test, label = "2PCR samples")
plt.scatter(pca3_train, pca3_test, label = "3PCR samples")
#plt.scatter([GP_train_av], [GP_test_av], label="GP average")
#plt.scatter([Ridge_train_av], [Ridge_test_av], label="Ridge average")
plt.xlabel("Training MAE")
plt.ylabel("Testing MAE")
plt.title('Comparative Errors')
plt.legend(loc='upper right')
plt.savefig('graphs/Comparison.png')
plt.show()

