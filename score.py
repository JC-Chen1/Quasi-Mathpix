import nltk
import distance
from nltk.translate.bleu_score import sentence_bleu

references = ["I love cats", "The sun is shining", "Hello, world!"]
hypotheses = ["I love dogs", "The moon is bright", "Hello, everyone!"]

#bilingual evaluation understudy score: 
def bleu_sore(references, hypotheses):
    bleu_score = 0.0
    for i,j in zip(references, hypotheses):
        bleu_score += max(sentence_bleu([i],j), 0.01)
    bleu_score = bleu_score/len(references) * 100
    return bleu_score

#edit distance
def edit_distance(references, hypotheses):
    '''Computes Levenshtein distance between two sequences.
    Args:
    references: list of sentences (one hypothesis)
    hypotheses: list of sentences (one hypothesis)
    Returns:
    1 - levenshtein distance: (higher is better, 1 is perfect)
    '''
    d_leven, len_tot = 0, 0
    for ref, hypo in zip(references, hypotheses):
        d_leven += distance.levenshtein(ref, hypo)
        len_tot += float(max(len(ref), len(hypo)))
    return (1. - d_leven/len_tot)*100

#exact math score:
#这里应该是需要预测的准确度进行计算
def exact_match_score(Accuracy):
    Exact_Match_Score = Accuracy * 100
    return Exact_Match_Score

blue_s = bleu_sore(references, hypotheses)
edit_d = edit_distance(references, hypotheses)
#ems =  exact_match_score(Accuracy)

print("BLEU Score: {:.2f}".format(blue_s))
print("Edit Distance: {:.2f}".format(edit_d))
#print("BLEU Score: {:.2f}".format(ems))
print("Overall Score: {:.2f}".format((blue_s + edit_d) / 2))