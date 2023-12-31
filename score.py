import nltk
import distance
from nltk.translate.bleu_score import sentence_bleu

# references = ["I love cats", "The sun is shining", "Hello, world!"]
# hypotheses = ["I love dogs", "The moon is bright", "Hello, everyone!"]

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
def exact_match_score(references, hypothesis):
    assert len(references) == len(hypothesis), 'references and hypothesis have different length!!!'
    count = 0
    for ref, hypo in zip(references, hypothesis):
        if ref == hypo:
            count += 1
    Accuracy = count / len(references)
    Exact_Match_Score = Accuracy * 100
    return Exact_Match_Score

def total_score(references, hypotheses):
    # 去除空格
    ref = [r.strip().replace(' ','') for r in references]
    hyp = [h.strip().replace(' ','') for h in hypotheses]
    print(f'debug: ref:{ref[0]}, {len(ref)}, hyp:{hyp[0]}, {len(ref)}')
    blue_s = bleu_sore(ref, hyp)
    edit_d = edit_distance(ref, hyp)
    exact_s = exact_match_score(ref, hyp)

    return blue_s, edit_d, exact_s, (blue_s + edit_d + exact_s) / 3


#ems =  exact_match_score(Accuracy)

# print("BLEU Score: {:.2f}".format(blue_s))
# print("Edit Distance: {:.2f}".format(edit_d))
# #print("BLEU Score: {:.2f}".format(ems))
# print("Overall Score: {:.2f}".format((blue_s + edit_d) / 2))