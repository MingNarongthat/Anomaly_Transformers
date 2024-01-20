from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

ref_caption = {
    "image1_id" :["soil fall down from the mountain and the grass field is on the cliff"]}
predicted_caption = {
    "image1_id" :["soil fall down from the cliff and the tree is on the cliff"]}

# Scoring
scorers = [
    (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
    (Meteor(),"METEOR"),
    (Rouge(), "ROUGE_L"),
    (Cider(), "CIDEr")
]
for scorer, method in scorers:
    print(f'Computing {method} score...')
    score, scores = scorer.compute_score(ref_caption, predicted_caption)
    if type(method) == list:
        for m, s in zip(method, score):
            print(f"{m}: {s}")
    else:
        print(f"{method}: {score}")


