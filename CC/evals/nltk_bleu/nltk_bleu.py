from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class NLTK_Bleu:
    def __init__(self):
        pass

    def compute_score(self, gts, res):

        smooth = SmoothingFunction()
        scores = [0, 0]

        for key in gts.keys():
            scores[0] += sentence_bleu(res[key], gts[key][0], smoothing_function=smooth.method1)
            scores[1] += 1

        return scores[0] / scores[1], 0

    def method(self):
        return "NLTK_Bleu"
