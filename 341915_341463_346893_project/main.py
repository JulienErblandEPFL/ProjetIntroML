import argparse
import random
import numpy as np
import matplotlib.pyplot as plt

from src.data import load_data
from src.methods.dummy_methods import DummyClassifier
from src.methods.logistic_regression import LogisticRegression
from src.methods.linear_regression import LinearRegression 
from src.methods.knn import KNN
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn
import os
np.random.seed(100)


def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end 
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data and flatten the images into vectors

    ##EXTRACTED FEATURES DATASET
    if args.data_type == "features":
        feature_data = np.load('features.npz',allow_pickle=True)
        xtrain, xtest, ytrain, ytest, ctrain, ctest =feature_data['xtrain'],feature_data['xtest'],\
        feature_data['ytrain'],feature_data['ytest'],feature_data['ctrain'],feature_data['ctest']

    ##ORIGINAL IMAGE DATASET (MS2)
    elif args.data_type == "original":
        data_dir = os.path.join(args.data_path,'dog-small-64')
        xtrain, xtest, ytrain, ytest, ctrain, ctest = load_data(data_dir)

    ##TODO: ctrain and ctest are for regression task. (To be used for Linear Regression and KNN)
    ##TODO: xtrain, xtest, ytrain, ytest are for classification task. (To be used for Logistic Regression and KNN)


    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.
    # Make a validation set (it can overwrite xtest, ytest)
    if not args.test:
        # Split the data into training and validation sets
        split_ratio = 0.8  # 80% training, 20% validation #ARBITRARY

        #mélange les datas
        size = len(xtrain)  #(=len(ytrain) = len(ctrain))
        pattern = [i for i in range(size)]
        random.shuffle(pattern)

        xtrain2 = np.empty_like(xtrain)
        ytrain2 = np.empty_like(ytrain)
        ctrain2 = np.empty_like(ctrain)

        for i, p in enumerate(pattern):
            xtrain2[p] = xtrain[i]
            ytrain2[p] = ytrain[i]
            ctrain2[p] = ctrain[i]

        #crée les validationSet
        xtest = xtrain2[int(len(xtrain) * split_ratio):]
        ytest = ytrain2[int(len(ytrain) * split_ratio):]
        ctest = ctrain2[int(len(ctrain) * split_ratio):]

        #crée les trainingSet
        xtrain = xtrain2[:int(len(xtrain) * split_ratio)]
        ytrain = ytrain2[:int(len(ytrain) * split_ratio)]
        ctrain = ctrain2[:int(len(ctrain) * split_ratio)]
    
    ### WRITE YOUR CODE HERE to do any other data processing
    
    #normalisation
    mu_train = np.mean(xtrain,0,keepdims=True)
    std_train = np.std(xtrain,0,keepdims=True)
    xtrain = normalize_fn(xtrain, mu_train, std_train)
    xtest = normalize_fn(xtest, mu_train, std_train)

    #biais appending
    xtrain = append_bias_term(xtrain)
    xtest = append_bias_term(xtest)



    ## 3. Initialize the method you want to use.

    # Use NN (FOR MS2!)
    if args.method == "nn":
        raise NotImplementedError("This will be useful for MS2.")

    # Follow the "DummyClassifier" example for your methods
    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)
    
    ### WRITE YOUR CODE HERE
    elif args.method == "linear_regression":
        method_obj = LinearRegression(lmda = args.lmda)
    elif args.method == "logistic_regression":
        method_obj = LogisticRegression(lr = args.lr, max_iters = args.max_iters)
    elif args.method == "knn":
        if args.task == "center_locating":
            method_obj = KNN(k = args.K, task_kind="regression", distance = args.distance, predict=args.predict)
        else :
            method_obj = KNN(k = args.K, task_kind="classification", distance = args.distance)
                


    ## 4. Train and evaluate the method
    if not args.plotting :
        print("Evaluating one example")
        if args.task == "center_locating":
            # Fit parameters on training data
            preds_train = method_obj.fit(xtrain, ctrain)

            # Perform inference for training and test data
            train_pred = method_obj.predict(xtrain)
            preds = method_obj.predict(xtest)

            ## Report results: performance on train and valid/test sets
            train_loss = mse_fn(train_pred, ctrain)
            loss = mse_fn(preds, ctest)

            print(f"\nTrain loss = {train_loss:.3f}% - Test loss = {loss:.3f}") #why is there "%" ???

        elif args.task == "breed_identifying":

            # Fit (:=train) the method on the training data for classification task
            preds_train = method_obj.fit(xtrain, ytrain)

            # Predict on unseen data
            preds = method_obj.predict(xtest)

            ## Report results: performance on train and valid/test sets
            acc = accuracy_fn(preds_train, ytrain)
            macrof1 = macrof1_fn(preds_train, ytrain)
            print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

            acc = accuracy_fn(preds, ytest)
            macrof1 = macrof1_fn(preds, ytest)
            print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
        else:
            raise Exception("Invalid choice of task! Only support center_locating and breed_identifying!")
    else :
        ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.
        print("Plotting")
        #lambdas = np.logspace(-3,1,num = 100,endpoint = True)
        iters = np.arange(100,1500,10)
        accuracy_1 = np.zeros(len(iters))

        data_train1 = xtrain
        data_train2 = ytrain
        data_test = xtest
        result_test = ytest

        for i in range(len(iters)):

            met = LogisticRegression(lr=0.0075,max_iters=iters[i])
            
            preds_train = met.fit(data_train1, data_train2)
            preds = met.predict(data_test)
            accuracy_1[i] = accuracy_fn(preds, result_test)


        best_it = iters[np.argmax(accuracy_1)]
        max_acc = np.max(accuracy_1)
        plt.scatter(best_it,max_acc,label = f"Best max_iter : max_iter = {best_it}, accuracy = {max_acc:.3f}%", color = 'r')
        # Tracer les données des deux tableaux
        plt.plot(iters, accuracy_1, label='Logistic regression',color = 'b')


        plt.grid(True)
        # Ajouter des étiquettes d'axe et une légende
    
        plt.xlabel('Number of iterations')
        plt.ylabel('Accuracy')
        plt.title('Relation between max_iter and accuracy(max_iter = 500)')
        plt.legend()

        # Afficher le graphe
        plt.show()



        

if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default="center_locating", type=str, help="center_locating / breed_identifying")
    parser.add_argument('--method', default="dummy_classifier", type=str, help="dummy_classifier / knn / linear_regression/ logistic_regression / nn (MS2)")
    parser.add_argument('--data_path', default="data", type=str, help="path to your dataset")
    parser.add_argument('--data_type', default="features", type=str, help="features/original(MS2)")
    parser.add_argument('--lmda', type=float, default=10, help="lambda of linear/ridge regression")
    parser.add_argument('--K', type=int, default=1, help="number of neighboring datapoints used for knn")
    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true", help="train on whole training data and evaluate on the test data, otherwise use a validation set")


    # Feel free to add more arguments here if you need!
    parser.add_argument('--distance', default= "euclidian",type = str, help="Methods to calculate the distance between two points : euclidian/chi-square")
    parser.add_argument('--predict', default= "average",type = str, help="Methods to approximate ŷ in kNN : average/weighted_average")
    parser.add_argument('--plotting',action="store_true", help="Executing the plot")

    # MS2 arguments
    parser.add_argument('--nn_type', default="cnn", help="which network to use, can be 'Transformer' or 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")

    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)
