from collections import defaultdict
from math import log, pow
import argparse

"""
Given a path to the corpus file, process it with our without unk for the models to access it
"""
class ProcessCorpus:
    corpusArray = []
    wordCountList = {}
    totalNumWords = 0
    unk = '<unk>'
    startSymbol = '<s>'
    stopSymbol = '</s>'

    """
    Initialialize class objects and process corpus file
    """
    def __init__(self, corpusPath, unk=False):
        self.corpusPath = corpusPath
        self.wordCountList = defaultdict(lambda: 0)
        self.totalNumWords = 0
        if unk:
            self.process_with_unk()
        else:
            self.process_default()

    """
    Process the corpus file, with unk symbols for rare words (count == 1)
    """
    def process_with_unk(self):
        f = open(self.corpusPath)
        self.corpusArray = []
        self.totalNumWords = 0
        self.wordCountList = defaultdict(lambda: 0)
        wordCountListHelper = defaultdict(lambda: 0)
        for sentence in f:
            words = sentence.split()
            for word in words:
                wordCountListHelper[word] = wordCountListHelper[word] + 1
                self.totalNumWords += 1

        rarewords = [key for key, count in wordCountListHelper.items() if count == 1]
        f.seek(0)
        for sentence in f:
            words = sentence.split()
            newsentence = []
            for word in words:
                if word in rarewords:
                    newsentence.append(self.unk)
                    self.wordCountList[self.unk] = self.wordCountList[self.unk] + 1
                else:
                    newsentence.append(word)
                    self.wordCountList[word] = self.wordCountList[word] + 1
            self.corpusArray.append(newsentence)

    """
    Process the corpus file as is
    """
    def process_default(self):
        f = open(self.corpusPath)
        self.corpusArray = []
        self.totalNumWords = 0
        self.wordCountList = defaultdict(lambda: 0)
        for sentence in f:
            sentence = sentence.strip()
            words = sentence.split()
            self.corpusArray.append(words)
            for word in words:
                self.wordCountList[word] = self.wordCountList[word] + 1
                self.totalNumWords += 1

"""
This class contains the unigram model for the subsequent interpolation 
"""
class UnigramModel:

    """
        Unigram model already gets the information it needs (word count) from the corpus data 
        which contains the ProcessCorpus object
    """
    def __init__(self, corpusData):
        self.trainingCorpus = corpusData

    """
    Scores the probability of a given sentence / word (if sentence contains one word
    """
    def score_probability_of_sentence(self, sentence):
        score = 0.0
        for word in sentence:
            wordcount = self.trainingCorpus.wordCountList[word]
            # log(a/b) = log(a) - log(b)
            if wordcount > 0:
                score += (log(wordcount, 2) - log(self.trainingCorpus.totalNumWords, 2))
            else:
                wordcount = self.trainingCorpus.wordCountList[ProcessCorpus.unk]
                score += (log(wordcount, 2) - log(self.trainingCorpus.totalNumWords, 2))

        return score

"""
This class represents the bigram model which would later be used for interpolation
"""
class BigramModel:

    def __init__(self, corpusData):
        self.trainingCorpus = corpusData
        self.bigramCountList = defaultdict(lambda: 0)
        self.train_bigram_model()

    """
    Splits corpus into bigrams for training the model as well as storing counts to be used later
    """
    def train_bigram_model(self):
        for sentence in self.trainingCorpus.corpusArray:
            unigram1 = ProcessCorpus.startSymbol
            self.trainingCorpus.wordCountList[unigram1] = self.trainingCorpus.wordCountList[unigram1] + 1
            unigram2 = ''
            for word in sentence:
                unigram2 = word
                self.bigramCountList[(unigram1,unigram2)] = self.bigramCountList[(unigram1,unigram2)] + 1
                unigram1 = word
            unigram2 = ProcessCorpus.stopSymbol
            self.bigramCountList[(unigram1, unigram2)] = self.bigramCountList[(unigram1, unigram2)] + 1

    """
    Scores the log probability of a given sentence using the information computed in the train function
    Using Laplace smoothing to prevent undefined probability in zero history situations
    """
    def score_probability_of_sentence(self, sentence):
        score = 0.0
        unigram1 = ProcessCorpus.startSymbol
        unigram2 = ''
        for word in sentence:
            unigram2 = word
            bigramFrequency = self.bigramCountList[(unigram1, unigram2)]
            #Used laplace smoothing #NOTE
            score += (log(bigramFrequency + 1, 2) - (log(self.trainingCorpus.wordCountList[unigram1] + len(self.trainingCorpus.wordCountList), 2)))
            unigram1 = word

        unigram2 = ProcessCorpus.stopSymbol
        bigramFrequency = self.bigramCountList[(unigram1, unigram2)]
        score += (log(bigramFrequency + 1, 2) - (log(self.trainingCorpus.wordCountList[unigram1] + len(self.trainingCorpus.wordCountList), 2)))

        return score

    """
    Score the MLE probability of a bigram from this model's trained data
    """
    def score_mle_probability(self, bigram):
        score = 0.0
        unigram1, unigram2 = bigram
        bigramFrequency = self.bigramCountList[bigram]
        # Used laplace smoothing here #NOTE
        score += (log(bigramFrequency + 1, 2) - (log(self.trainingCorpus.wordCountList[unigram1] + len(self.trainingCorpus.wordCountList), 2)))
        return score

"""
This class represents the trigram model which would later be used for interpolation
"""
class TrigramModel:

    def __init__(self, corpusData, relatedBigram, delta=0):
        self.trainingCorpus = corpusData
        self.trigramCountList = defaultdict(lambda: 0)
        self.delta = delta
        self.relatedBigram = relatedBigram
        self.train_trigram_model()

    """
    Splits corpus into trigram for training the model as well as storing counts to be used later
    """
    def train_trigram_model(self):
        for sentence in self.trainingCorpus.corpusArray:
            unigram1 = ProcessCorpus.startSymbol
            unigram2 = ProcessCorpus.startSymbol
            unigram3 = ''
            self.relatedBigram.bigramCountList[(unigram1, unigram2)] = self.relatedBigram.bigramCountList[(unigram1, unigram2)] + 1
            for word in sentence:
                unigram3 = word
                self.trigramCountList[(unigram1, unigram2, unigram3)] = self.trigramCountList[(unigram1, unigram2, unigram3)] + 1
                unigram1 = unigram2
                unigram2 = word
            unigram3 = ProcessCorpus.stopSymbol
            self.trigramCountList[(unigram1, unigram2, unigram3)] = self.trigramCountList[(unigram1, unigram2, unigram3)] + 1

    """
    Scores the log probability of a given sentence using the information computed in the train function
    Using Laplace smoothing to prevent undefined probability in zero history situations
    """
    def score_probability_of_sentence(self, sentence):
        score = 0.0
        unigram1 = ProcessCorpus.startSymbol
        unigram2 = ProcessCorpus.startSymbol
        unigram3 = ''
        for word in sentence:
            unigram3 = word
            trigramFrequency = self.trigramCountList[(unigram1, unigram2, unigram3)]
            bigramFrequency = self.relatedBigram.bigramCountList[(unigram1, unigram2)]
            #Used laplace smoothing #NOTE
            score += (log((trigramFrequency + 1) - self.delta, 2) - (log(bigramFrequency + len(self.trainingCorpus.wordCountList), 2)))
            unigram1 = unigram2
            unigram2 = word

        unigram3 = ProcessCorpus.stopSymbol
        trigramFrequency = self.trigramCountList[(unigram1, unigram2, unigram3)]
        bigramFrequency = self.relatedBigram.bigramCountList[(unigram1, unigram2)]
        score += (log((trigramFrequency + 1) - self.delta, 2) - (log(bigramFrequency + len(self.trainingCorpus.wordCountList), 2)))
        return score

    """
    Scores the MLE probability of a given trigram from the trained data
    """
    def score_mle_probability(self, trigram):
        score = 0.0
        unigram1, unigram2, unigram3 = trigram

        trigramFrequency = self.trigramCountList[trigram]
        bigramFrequency = self.relatedBigram.bigramCountList[(unigram1, unigram2)]
        #Used laplace smoothing here #NOTE
        score += (log((trigramFrequency + 1) - self.delta, 2) - (log(bigramFrequency + len(self.trainingCorpus.wordCountList), 2)))

        return score

"""
This model interpolates a unigram, bigram and trigram model with some hyperparameters as weights for each model
"""
class InterpolationModel:

        """
        initialize individual models with training done in initialization
        """
        def __init__(self, corpusData, uniweight=0.2, biweight=0.3, triweight=0.5):
            self.uniweight = uniweight
            self.biweight = biweight
            self.triweight = triweight
            self.trainingData = corpusData
            self.unigramModel = UnigramModel(corpusData)
            self.bigramModel = BigramModel(corpusData)
            self.trigramModel = TrigramModel(corpusData, self.bigramModel, 0)

        """
        Score a sentence with the interpolation of the three models and weights
        """
        def score_sentence(self, sentence):
            score = 0.0
            score += self.uniweight * self.unigramModel.score_probability_of_sentence(sentence)
            score += self.biweight * self.bigramModel.score_probability_of_sentence(sentence)
            score += self.triweight * self.trigramModel.score_probability_of_sentence(sentence)
            return score

        """
           Calculate perplexity of a corpus for this  model
        """
        def calculate_perplexity(self, corpus):
            logSum = 0.0
            numWordsInCorpus = 0
            perplexity = 0.0
            for sentence in corpus.corpusArray:
                numWordsInCorpus += len(sentence)
                logSum += (-1 * self.score_sentence(sentence))

            perplexity = logSum / numWordsInCorpus
            perplexity = pow(2, perplexity)
            return perplexity

"""
This model creates the backoff model implementation for the proposed modification to Assignment 1
"""
class BackoffModel:

    """
    Initializes with raw corpus data or an existing set of trained unigram, bigram and trigram models
    """
    def __init__(self, corpusData, delta, unigramModel = None, bigramModel = None, trigramModel = None):
        self.trainingCorpus = corpusData
        if unigramModel is not None:
            self.unigramModel = unigramModel
        else:
            self.unigramModel = UnigramModel(corpusData)

        if bigramModel is not None:
            self.bigramModel = bigramModel
        else:
            self.bigramModel = BigramModel(corpusData)

        if trigramModel is not None:
            self.trigramModel = trigramModel
            self.trigramModel.delta = delta
        else:
            self.trigramModel = TrigramModel(corpusData, self.bigramModel, delta)

        self.historyList = defaultdict()  #for each history contains B(w_i-2, w_i-1) and #B(w_i-1)
        self.B1List = defaultdict() #for each bigram contains B(w_i-1)
        self.train_model()

    """
    Train model by computing additional data to help with 'missing mass' value 
    """
    def train_model(self):
        for history, count in self.bigramModel.bigramCountList.items():
            newHistory = BackoffData(history)
            unigram1, unigram2 = history
            newB1History = BackoffData(unigram2)
            totalprob = 0.0
            for word, freq in self.trainingCorpus.wordCountList.items():
                if self.trigramModel.trigramCountList.get((unigram1, unigram2, word), 0) > 0:
                    totalprob += self.trigramModel.score_mle_probability((unigram1, unigram2, word))
                else:
                    newHistory.B2.append(word)
                    if self.bigramModel.bigramCountList.get((unigram2, word), 0) == 0:
                        newHistory.B1.append(word)
                        newB1History.B1.append(word)
            newHistory.q = 1 - (pow(2, totalprob))
            newB1History.q = newHistory.q
            self.historyList[history] = newHistory
            self.B1List[unigram2] = newB1History

    """
    Score a sentence in this model 
    """
    def score_sentence(self, sentence):
        score = 0.0
        unigram1 = ProcessCorpus.startSymbol
        unigram2 = ProcessCorpus.startSymbol
        unigram3 = ''
        for word in sentence:
            unigram3 = word
            trigram = (unigram1, unigram2, unigram3)
            bigram = (unigram2, unigram3)
            if self.trigramModel.trigramCountList[trigram] > 0:
                #p1 case
                score += self.trigramModel.score_mle_probability(trigram)
            elif (self.trigramModel.trigramCountList[trigram] == 0) and (self.bigramModel.bigramCountList[bigram] > 0):
                #p2 case
                numerator = self.bigramModel.score_mle_probability(bigram)
                denominator = 0.0
                historyitem = [historydata for historykey, historydata in self.historyList.items() if historykey == bigram]
                historyitem = historyitem[0]
                for b2item in historyitem.B2:
                    denominator += self.bigramModel.score_mle_probability((unigram2, b2item))
                score += (numerator/denominator) * 0.5 * historyitem.q
            elif self.bigramModel.bigramCountList[bigram] == 0:
                #p3 case
                numerator = self.unigramModel.score_probability_of_sentence([unigram3])
                denominator = 0.0
                historyitem = [historydata for historykey, historydata in self.historyList.items() if historykey == bigram]
                if len(historyitem) > 0:
                    historyitem = historyitem[0]
                    for b1item in historyitem.B1:
                        denominator += self.unigramModel.score_probability_of_sentence([b1item])
                    score += (numerator / denominator) * 0.5 * historyitem.q
                else:
                    historyitem = [historydata for historykey, historydata in self.B1List.items() if historykey == unigram2]
                    if len(historyitem) > 0:
                        historyitem = historyitem[0]
                        for b1item in historyitem.B1:
                            denominator += self.unigramModel.score_probability_of_sentence([b1item])
                        score += (numerator / denominator) * 0.5 * historyitem.q
                    else:
                        # unknown word for unigram calculation
                        historyitem = [historydata for historykey, historydata in self.B1List.items() if historykey == ProcessCorpus.unk]
                        if len(historyitem) > 0:
                            historyitem = historyitem[0]
                            for b1item in historyitem.B1:
                                denominator += self.unigramModel.score_probability_of_sentence([b1item])
                            score += (numerator / denominator) * 0.5 * historyitem.q

            unigram1 = unigram2
            unigram2 = word

        unigram3 = ProcessCorpus.stopSymbol
        trigram = (unigram1, unigram2, unigram3)
        bigram = (unigram2, unigram3)
        if self.trigramModel.trigramCountList[trigram] > 0:
            # p1 case
            score += self.trigramModel.score_mle_probability(trigram)
        elif (self.trigramModel.trigramCountList[trigram] == 0) and (self.bigramModel.bigramCountList[bigram] > 0):
            # p2 case
            numerator = self.bigramModel.score_mle_probability(bigram)
            denominator = 0.0
            historyitem = [historydata for historykey, historydata in self.historyList.items() if historykey == bigram]
            historyitem = historyitem[0]
            for b2item in historyitem.B2:
                denominator += self.bigramModel.score_mle_probability((unigram2, b2item))
            score += (numerator / denominator) * 0.5 * historyitem.q
        elif self.bigramModel.bigramCountList[bigram] == 0:
            # p3 case
            numerator = self.unigramModel.score_probability_of_sentence([unigram3])
            denominator = 0.0
            historyitem = [historydata for historykey, historydata in self.historyList.items() if historykey == bigram]
            if len(historyitem) > 0:
                historyitem = historyitem[0]
                for b1item in historyitem.B1:
                    denominator += self.unigramModel.score_probability_of_sentence([b1item])
                score += (numerator / denominator) * 0.5 * historyitem.q
            else:
                historyitem = [historydata for historykey, historydata in self.B1List.items() if historykey == unigram2]
                if len(historyitem) > 0:
                    historyitem = historyitem[0]
                    for b1item in historyitem.B1:
                        denominator += self.unigramModel.score_probability_of_sentence([b1item])
                    score += (numerator / denominator) * 0.5 * historyitem.q
                else:
                    #unknown word for unigram calculation
                    historyitem = [historydata for historykey, historydata in self.B1List.items() if historykey == ProcessCorpus.unk]
                    if len(historyitem) > 0:
                        historyitem = historyitem[0]
                        for b1item in historyitem.B1:
                            denominator += self.unigramModel.score_probability_of_sentence([b1item])
                        score += (numerator / denominator) * 0.5 * historyitem.q

        return score

    """
    Calculate perplexity of a corpus for this  model
    """
    def calculate_perplexity(self, corpus):
        logSum = 0.0
        numWordsInCorpus = 0
        perplexity = 0.0
        for sentence in corpus.corpusArray:
            numWordsInCorpus += len(sentence)
            logSum += (-1 * self.score_sentence(sentence))

        perplexity = logSum / numWordsInCorpus
        perplexity = pow(2, perplexity)
        return perplexity

"""
This holds the data for each history in backoff model
"""
class BackoffData:

    def __init__(self, history):
        self.history = history #history
        self.q = 0.0 #missing mass
        self.B2 = [] #B(w_i-2, w_i-1)
        self.B1 = [] #B(w_i-1)
        #TODO:store this data into file once computed

"""
Parge command line arguments: training file path and test file path for a model
"""
def parse_args():
    argParser = argparse.ArgumentParser(description='Parse settings to run models')
    argParser.add_argument('filepathtrain', help='Path to file to train model')
    argParser.add_argument('filepathtest', help='Path to file to test model')
    options = argParser.parse_args()
    return options

"""
Execute an instance of interpolation and backoff model for a given train and test corpus
"""
def main():

    args = parse_args()
    trainDataPath = args.filepathtrain
    testDataPath = args.filepathtest

    corpusData = ProcessCorpus(trainDataPath, True)

    interpolationModel = InterpolationModel(corpusData, 0.05, 0.15, 0.8)
    testCorpus = ProcessCorpus(testDataPath, False)
    interpolation_perplexity = interpolationModel.calculate_perplexity(testCorpus)

    backoffModel = BackoffModel(corpusData, 0.5, interpolationModel.unigramModel, interpolationModel.bigramModel, interpolationModel.trigramModel)
    backoff_perplexity = backoffModel.calculate_perplexity(testCorpus)

    with open("outputData.txt", 'w') as f:
        f.write("Trainfile: " + trainDataPath + '\r\n')
        f.write("Testpath: " + testDataPath + '\r\n')
        f.write("Interpolation Perplexity: " + str(interpolation_perplexity) + '\r\n')
        f.write("Backoff Perplexity: " + str(backoff_perplexity) + '\r\n')
        f.write("Interpolation hyperparameters: 0.05, 0.15, 0.8" + '\r\n')
        f.write("Backoff delta hyperparameters: 0.5" + '\r\n')

if __name__ == '__main__':
    main()