from sklearn.metrics.ranking import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score,multilabel_confusion_matrix
from keras.layers import Dense, Dropout, Activation
from keras.layers import BatchNormalization
from imblearn.keras import balanced_batch_generator
from sklearn.metrics import cohen_kappa_score
from imblearn.under_sampling import NearMiss
from keras.models import Sequential
from keras.layers import Conv2D
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shap
import tqdm
import os


def myRound(y, number):
    y_ = []
    for i in y:
        if i >= number:
            y_.append(1.)
        else:
            y_.append(0.)
    return np.array(y_)



class Trainer:

    """Manages the training of deep neural networks. For each species five models are trained, and the model which has
    scored the highest AUC value on the testing dataset is subsequently saved to file. Using a random subset of the
    test set the impact of the environmental layers is estimated and saved to an image file. The class also creates an
    evaluation file containing various (average) performance metrics per species like accuracy, loss, AUC, etc.

    :param oh: an Occurrence object: holds occurrence files and tables
    :param gh: a GIS object: holds path and file names required for computation of gis data.
    :param ch: a Config object: holds instance variables that determines the random seed, batch size, number of epochs,
    number of nodes in each model layers and dropout for each model layer.
    :param verbose: a boolean: prints a progress bar if True, silent if False

    :return: Object. Used to train five models, and save the one with the highest AUC score to file.
    Performed by calling class method train on Trainer object.
    """

    def __init__(self, oh, gh, ch, verbose):

        self.oh = oh
        self.gh = gh
        self.ch = ch
        self.verbose = verbose

        self.spec = ''

        self.variables = []

        self.test_loss = []
        self.test_acc = []
        self.test_AUC = []
        self.test_tpr = []
        self.test_uci = []
        self.test_lci = []
        self.sensitivity = []
        self.tss = []
        self.kappa = []
        self.bias = []
        self.prevalence = []
        self.best_model_auc = [0]

        self.occ_len = 0
        self.abs_len = 0

        self.random_seed = self.ch.random_seed

        self.batch = self.ch.batchsize
        self.epoch = self.ch.epoch

        self.model_layers = self.ch.model_layers
        self.model_dropout = self.ch.model_dropout

    def mcmsave(self,y_true, y_pred, count):
        mcm = multilabel_confusion_matrix(myRound(y_true, count), myRound(y_pred, count))
        tp = mcm[:, 1, 1][1]  # a
        fn = mcm[:, 1, 0][1]  # b
        fp = mcm[:, 0, 1][1]  # c
        tn = mcm[:, 0, 0][1]  # d

        print("tp:", tp)
        print("fn:", fn)
        print("fp:", fp)
        print("tn:", tn)

        print("Sensitivity :", (tp / (tp + fp)))  # Sensitivity
        Sensitivity = ((tp / (tp + fp)))
        # print((tn / (fp + tn))[1]) #
        Sorensen = (2*tp) / (fn + (2*tp) + fp)
        OPR = fp / (tp + fp)
        UPR = fn / (tp + fn)
        print(Sorensen)
        print(OPR)
        print(UPR)
        print("TSS :", (tp / (tp + fp)) + (tn / (fn + tn)) - 1)
        print("KAPPA :", cohen_kappa_score(myRound(y_pred, count), myRound(y_true, count)))
        print("bias :", (((tp + fn) / (tp + fp))))
        print("prevalence :", ((tp + fp) / (tp + fp + fn + tn)))
        tss = ((tp / (tp + fp)) + (tn / (fn + tn)) - 1)
        kappa = (cohen_kappa_score(myRound(y_pred, count), myRound(y_true, count)))
        bias = (((tp + fn) / (tp + fp)))
        prevalence = (((tp + fp) / (tp + fp + fn + tn)))

        with open(self.ch.result_path + '/_DNN_performance/result_'+str(count)+'.txt', 'a') as file2:
            file2.write(self.spec + "\t" + str(tss) + "\t" + str(Sensitivity) + "\t" + str(kappa) + "\t" + str(
                bias) + "\t" + str(prevalence) + "\n")
            file2.close()



    def create_eval(self):

        """Creates a new eval file containing a basic column layout.

        :return: None. Writes column layout to evaluation file. Overwrites previous evaluation file if present.
        """

        if not os.path.isdir(self.ch.result_path + '/_DNN_performance'):
            os.makedirs(self.ch.result_path + '/_DNN_performance')

        with open(self.ch.result_path + '/_DNN_performance/DNN_eval.txt', 'w+') as file:
            file.write(
                "Species" + "\t" + "Test_loss" + "\t" + "Test_acc" + "\t" + "Test_tpr" + "\t" + "Test_AUC" + "\t" + "Test_LCI95%" + "\t" + "Test_UCI95%" + "\t" + "occ_samples" + "\t" + "abs_samples" + "\t"+ "tss" + "\t"+ "sensitivity" + "\t"+ "kappa" + "\t"+ "bias" + "\t"+ "prevalence" + "\n")
            file.close()

        with open(self.ch.result_path + '/_DNN_performance/pred.txt', 'w+') as file1:
            file1.write("")
            file1.close()

    def create_input_data(self):

        """Loads data from file and performs final data preparations before training. Returns all input data for
        training the model and determining variable importance.

        :return: Tuple. Containing:
        array 'X' an array containing all occurrences for a certain species;
        array 'X_train' an array containing all training data for a certain species;
        array 'X_test' an array containing all testing data for a certain species;
        array 'y_train' an array holding all the labels (ground truth, in this case absent=0 / present=1) for the
        training set;
        array 'y_test' an array holding all the labels for the test set;
        table 'test_set' pandas dataframe containing a copy of array 'X_test';
        array 'shuffled_X_train' an array containing a random subset of the X_train data;
        array 'shuffled_X_test' an array containing a random subset of the X_test data.
        """

        np.random.seed(self.random_seed)
        self.variables = self.gh.names.copy()
        # print("%s_presence_map" % self.spec)
        self.variables.remove("%s_presence_map" % self.spec)

        table = pd.read_csv(self.gh.spec_ppa_env + '/%s_env_dataframe_train.csv' % self.spec)

        table = table.drop('%s_presence_map' % self.spec, axis=1)
        table = table.dropna(axis=0, how="any")
        band_columns = [column for column in table.columns[1:len(self.gh.names)]]
        X = []
        y = []
        for _, row in table.iterrows():
            x = row[band_columns].values
            x = x.tolist()
            x.append(row["present/pseudo_absent"])
            X.append(x)
        df = pd.DataFrame(data=X, columns=band_columns + ["presence"])
        df.to_csv(self.gh.root + '/filtered.csv', index=None)
        self.occ_len = int(len(df[df["presence"] == 1]))
        self.abs_len = int(len(df[df["presence"] == 0]))
        X = []
        y = []
        band_columns = [column for column in df.columns[:-1]]
        for _, row in df.iterrows():
            X.append(row[band_columns].values.tolist())
            y.append([1 - row["presence"], row["presence"]])
        X_train = np.vstack(X)
        y_train = np.vstack(y)

        table = pd.read_csv(self.gh.spec_ppa_env + '/%s_env_dataframe_test.csv' % self.spec)
        print(len(table))
        zero_data = np.zeros(shape=(len(table)))
        self.total_pred = pd.DataFrame(zero_data)
        table = table.drop('%s_presence_map' % self.spec, axis=1)
        table = table.dropna(axis=0, how="any")
        band_columns = [column for column in table.columns[1:len(self.gh.names)]]
        X = []
        y = []
        for _, row in table.iterrows():
            x = row[band_columns].values
            x = x.tolist()
            x.append(row["present/pseudo_absent"])
            X.append(x)
        df = pd.DataFrame(data=X, columns=band_columns + ["presence"])
        df.to_csv(self.gh.root + '/filtered.csv', index=None)
        self.occ_len = int(len(df[df["presence"] == 1]))
        self.abs_len = int(len(df[df["presence"] == 0]))
        X = []
        y = []
        band_columns = [column for column in df.columns[:-1]]
        for _, row in df.iterrows():
            X.append(row[band_columns].values.tolist())
            y.append([1 - row["presence"], row["presence"]])
        X_test = np.vstack(X)
        y_test = np.vstack(y)


        X=np.vstack((X_train,X_test))
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=self.random_seed)
        test_set = pd.DataFrame(X_test)
        test_set.rename(columns=dict(zip(test_set.columns[0:len(self.gh.names) - 1], self.variables)), inplace=True)
        shuffled_X_train = X_train.copy()
        np.random.shuffle(shuffled_X_train)
        shuffled_X_train = shuffled_X_train[:1000]
        shuffled_X_test = X_test.copy()
        np.random.shuffle(shuffled_X_test)
        shuffled_X_test = shuffled_X_test[:1000]
        return X, X_train, X_test, y_train, y_test, test_set, shuffled_X_train, shuffled_X_test

    def create_model_architecture(self, X):

        """Creates a model architecture based on instance variables 'model_layers' and 'models_dropout'. Default is a
        model with 4 layers [250, 200, 150 and 100 nodes per layer] with dropout [0.5, 0.3, 0.5, 0.3 per layer]

        :param X: Array. used to define the input dimensions (number of nodes in the first layer) of the model.
        :return: Keras Model Object. Initialized model with a specific architecture ready for training.
        """

        num_classes = 2
        num_inputs = X.shape[1]
        model = Sequential()
        if len(self.model_layers) == len(self.model_dropout):
            for l in range(len(self.model_layers)):
                layer = Dense(self.model_layers[l], activation='relu', input_shape=(num_inputs,))
                model.add(layer)
                model.add(Dropout(self.model_dropout[l]))

        out_layer = Dense(num_classes, activation=None)
        model.add(out_layer)
        model.add(Activation("softmax"))
        model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model


    def train_model(self, model, X_train, X_test, y_train, y_test):

        """Training a model to predict the presence or absence of a species. Various instance variables are used to
        define how the model trains, like: batch size, random seed and number of epochs.

        :param model: Keras Model Object. Initialized model ready for training.
        :param X_train: Array. Contains training data.
        :param X_test: Array. Contains testing data.
        :param y_train: Array. Contains training (ground truth) labels.
        :param y_test: Array. Contains testing (ground truth) labels.

        :return: Tuple. Containing:
        float 'AUC' performance metric between 0 and 1 (0 = 100% wrong, 1 = 100% right);
        keras model 'model' a keras model with an identical architecture to the input variable 'model' but with trained
        weights.
        """

        training_generator, steps_per_epoch = balanced_batch_generator(X_train, y_train, sampler=NearMiss(),
                                                                       batch_size=self.batch, random_state=self.random_seed)
        hist = model.fit_generator(generator=training_generator, steps_per_epoch=steps_per_epoch, epochs=self.epoch,
                            verbose=0)

        # print(hist.history['loss'])
        # print(hist.history['acc'])

        score = model.evaluate(X_test, y_test, verbose=0)
        predictions = model.predict(X_test)

        y_pred = predictions[:, 1]

        fpr, tpr, thresholds = roc_curve(y_test[:, 1], y_pred)
        len_tpr = int(len(tpr) / 2)
        self.test_loss.append(score[0])
        self.test_acc.append(score[1])
        self.test_AUC.append(roc_auc_score(y_test[:, 1], predictions[:, 1]))
        self.test_tpr.append(tpr[len_tpr])

        AUC = roc_auc_score(y_test[:, 1], y_pred)

        n_bootstraps = 1000
        y_pred_pd=pd.DataFrame((y_pred))
        self.total_pred=pd.concat([self.total_pred+y_pred_pd],axis=1)
        print( self.total_pred)
        # print(len(y_pred))
        # print("예측:",np.round(y_pred, 0))

        y_true = y_test[:, 1]
        # print("실제:",np.round(y_true, 0))

        mcm = multilabel_confusion_matrix(np.round(y_true, 0), np.round(y_pred, 0))
        self.mcmsave(y_true,y_pred,0.1)
        self.mcmsave(y_true, y_pred, 0.2)
        self.mcmsave(y_true, y_pred, 0.3)
        self.mcmsave(y_true, y_pred, 0.4)
        self.mcmsave(y_true, y_pred, 0.5)
        self.mcmsave(y_true, y_pred, 0.6)
        self.mcmsave(y_true, y_pred, 0.7)
        self.mcmsave(y_true, y_pred, 0.8)
        self.mcmsave(y_true, y_pred, 0.9)

        tp = mcm[:, 1, 1][1]  # a
        fn = mcm[:, 1, 0][1]  # b
        fp = mcm[:, 0, 1][1]  # c
        tn = mcm[:, 0, 0][1]  # d

        print("tp:",tp)
        print("fn:",fn)
        print("fp:",fp)
        print("tn:",tn)

        print("auc :",roc_auc_score(y_test[:, 1], y_pred))
        print("Sensitivity :",(tp / (tp + fp))) # Sensitivity
        self.sensitivity.append((tp / (tp + fp)))
        # print((tn / (fp + tn))[1]) #
        print("TSS :",(tp / (tp + fp)) + (tn / (fn + tn)) - 1)
        print("KAPPA :",cohen_kappa_score(np.round(y_pred, 0),np.round(y_true, 0)))
        print("bias :",(((tp+fn)/(tp+fp))))
        print("prevalence :",((tp+fp)/(tp+fp+fn+tn)))
        self.tss.append((tp / (tp + fp)) + (tn / (fn + tn)) - 1)
        self.kappa.append(cohen_kappa_score(np.round(y_pred, 0),np.round(y_true, 0)))
        self.bias.append(((tp+fn)/(tp+fp)))
        self.prevalence.append(((tp+fp)/(tp+fp+fn+tn)))
        bootstrapped_scores = []
        rng = np.random.RandomState(self.random_seed)
        for i in range(n_bootstraps):
            indices = rng.randint(0, len(y_pred) - 1, len(y_pred))
            if len(np.unique(y_true[indices])) < 2:
                continue
            score = roc_auc_score(y_true[indices], y_pred[indices])
            bootstrapped_scores.append(score)
        sorted_scores = np.array(bootstrapped_scores)
        sorted_scores.sort()
        ci_lower = sorted_scores[int(0.05 * len(sorted_scores))]
        ci_upper = sorted_scores[int(0.95 * len(sorted_scores))]
        self.test_lci.append(ci_lower)
        self.test_uci.append(ci_upper)
        return AUC, model

    def validate_model(self, model, AUC, X_train, X_test, shuffled_X_train, shuffled_X_test, test_set):

        """Validate the model based on the AUC score. If the current models AUC is higher then previous version(s) of
        the model is saved to file, and the feature importance for the current model is calculated, and subsequently
        saved to image.

        :param model: Keras Model. Model object trained using the class method train_model.
        :param AUC: Float. Performance metric with a score between 0 and 1 (0 = 100% wrong, 1 = 100% right).
        :param X_train: Array. Contains training data.
        :param X_test: Array. Contains testing data.
        :param shuffled_X_train: an array containing a random subset of the X_train data;
        :param shuffled_X_test:an array containing a random subset of the X_test data;
        :param test_set: pandas dataframe containing a copy of array 'X_test';

        :return: None. Does not return value or object, instead saves a model to file if the AUC score is higher than
        previous versions. If a model is saved to file the feature importance is tested and also saved to an image file.
        """

        if AUC > self.best_model_auc[0]:
            if not os.path.isdir(self.ch.result_path + '/%s' % self.spec):
                os.makedirs(self.ch.result_path + '/%s' % self.spec, exist_ok=True)
            self.best_model_auc[0] = AUC
            model_json = model.to_json()
            with open(self.ch.result_path + '/{}/{}_model.json'.format(self.spec, self.spec), 'w') as json_file:
                json_file.write(model_json)
            model.save_weights(self.ch.result_path + '/{}/{}_model.h5'.format(self.spec, self.spec))
            if int(len(X_train)) > 5000:
                explainer = shap.DeepExplainer(model, shuffled_X_train)
                test_set = pd.DataFrame(shuffled_X_test)
                test_set.rename(columns=dict(zip(test_set.columns[0:len(self.gh.names) - 1], self.variables)),
                                inplace=True)
                shap_values = explainer.shap_values(shuffled_X_test)
                shap.summary_plot(shap_values[1], test_set, show=False)
                plt.savefig(self.ch.result_path + '/{}/{}_feature_impact'.format(self.spec, self.spec),
                            bbox_inches="tight")
                plt.close()
            else:
                explainer = shap.DeepExplainer(model, X_train)
                shap_values = explainer.shap_values(X_test)
                shap.summary_plot(shap_values[1], test_set, show=False)
                plt.savefig(self.ch.result_path + '/{}/{}_feature_impact'.format(self.spec, self.spec),
                            bbox_inches="tight")
                plt.close()

    def update_performance_metrics(self):

        """Updates the evaluation file created at the start of the training process. For each species adds one new line
        to the evaluation document. For each species a number of average performance metrics are computed, such as:
        testing loss, testing accuracy, AUC score, True positive rate, lower confidence interval and upper
        confidence interval.

        :return: None. Does not return value or object, instead appends one line of performance metrics for each species
        to the initially created evaluation file.
        """

        avg_loss = sum(self.test_loss) / len(self.test_loss)
        avg_acc = sum(self.test_acc) / len(self.test_acc)
        avg_AUC = sum(self.test_AUC) / len(self.test_AUC)
        avg_tpr = sum(self.test_tpr) / len(self.test_tpr)
        avg_lci = sum(self.test_lci) / len(self.test_lci)
        avg_uci = sum(self.test_uci) / len(self.test_uci)
        avg_tss = sum(self.tss)/len(self.tss)
        avg_kappa = sum(self.kappa) / len(self.kappa)
        avg_bias = sum(self.bias) / len(self.bias)
        avg_prevalence = sum(self.prevalence) / len(self.prevalence)
        avg_sensitivity = sum(self.sensitivity)/len(self.sensitivity)

        with open(self.ch.result_path + '/_DNN_performance/DNN_eval.txt', 'a') as file:
            file.write(self.spec + "\t" + str(avg_loss) + "\t" + str(avg_acc) + "\t" + str(avg_tpr) + "\t" + str(
                avg_AUC) + "\t" + str(avg_lci) + "\t" + str(avg_uci) + "\t" + str(self.occ_len) + "\t" + str(
                self.abs_len) + "\t"+ str(avg_tss) + "\t"+ str(avg_sensitivity) +"\t"+ str(avg_kappa)+"\t"+ str(avg_bias) +"\t"+ str(avg_prevalence) + "\n")
            file.close()

        with open(self.ch.result_path + '/_DNN_performance/pred.txt', 'a') as file2:
            np.savetxt(file2, self.total_pred/5)
            file2.close()

    def train(self):

        """Training manager that performs the entire training procedure from start to finish. Responsible for:
        Creating an evaluation file, obtaining the data, creating models, training models and validating their
        performance, saving models to file and updating the performance metrics.

        :return: None. Does not return value or object, instead writes the resulting models to files, writes performance
        metrics to file and computing the feature importance of the input variables.
        """

        self.create_eval()

        for self.spec in tqdm.tqdm(self.oh.name, desc='training models' + (35 * ' '), leave=True) if self.verbose else self.oh.name:
            X, X_train, X_test, y_train, y_test, test_set, shuffled_X_train, shuffled_X_test = self.create_input_data()
            self.test_loss = []
            self.test_acc = []
            self.test_AUC = []
            self.test_tpr = []
            self.test_lci = []
            self.test_uci = []
            self.sensitivity=[]
            self.tss=[]
            self.bias = []
            self.prevalence = []
            self.best_model_auc = [0]
            for i in (tqdm.tqdm(range(1, 6), desc='%s' % self.spec + ((50 - len(self.spec)) * ' '), leave=True) if self.verbose else range(1, 6)):
                model = self.create_model_architecture(X)
                AUC, model = self.train_model(model, X_train, X_test, y_train, y_test)
                self.validate_model(model, AUC, X_train, X_test, shuffled_X_train, shuffled_X_test, test_set)
            self.update_performance_metrics()