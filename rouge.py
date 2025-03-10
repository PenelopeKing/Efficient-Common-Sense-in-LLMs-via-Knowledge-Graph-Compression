import numpy as np

def my_lcs(string, sub):
    """
    Calculates longest common subsequence for two token lists.
    :param string: list of str – tokens from a string (split by whitespace)
    :param sub: list of str – tokens from the other string
    :returns: int – length of the longest common subsequence
    """
    if len(string) < len(sub):
        sub, string = string, sub

    lengths = [[0 for _ in range(len(sub)+1)] for _ in range(len(string)+1)]

    for j in range(1, len(sub)+1):
        for i in range(1, len(string)+1):
            if string[i-1] == sub[j-1]:
                lengths[i][j] = lengths[i-1][j-1] + 1
            else:
                lengths[i][j] = max(lengths[i-1][j], lengths[i][j-1])
    return lengths[len(string)][len(sub)]

class Rouge:
    """
    Class for computing ROUGE-L score.
    """
    def __init__(self):
        # Parameter beta from common ROUGE-L definition.
        self.beta = 1.2

    def calc_score(self, candidate, refs):
        """
        Compute ROUGE-L score given one candidate and multiple references.
        :param candidate: list containing a single candidate sentence (str)
        :param refs: list of reference sentences (str)
        :returns: float – ROUGE-L score
        """
        assert len(candidate) == 1
        assert len(refs) > 0         
        prec = []
        rec = []
        # Tokenize candidate sentence
        token_c = candidate[0].split(" ")
        for reference in refs:
            token_r = reference.split(" ")
            lcs = my_lcs(token_r, token_c)
            prec.append(lcs / float(len(token_c)))
            rec.append(lcs / float(len(token_r)))
        prec_max = max(prec)
        rec_max = max(rec)
        if prec_max != 0 and rec_max != 0:
            score = ((1 + self.beta**2) * prec_max * rec_max) / (rec_max + self.beta**2 * prec_max)
        else:
            score = 0.0
        return score

    def compute_score(self, gts, res):
        """
        Computes the average ROUGE-L score.
        :param gts: dict mapping example IDs to reference sentences (list of str)
        :param res: dict mapping example IDs to candidate sentence (list of one str)
        :returns: (float, np.array) – average score and per-example scores
        """
        assert gts.keys() == res.keys()
        scores = []
        for id in gts.keys():
            hypo = res[id]
            ref  = gts[id]
            scores.append(self.calc_score(hypo, ref))
            # Sanity checks:
            assert isinstance(hypo, list) and len(hypo) == 1
            assert isinstance(ref, list) and len(ref) > 0
        average_score = np.mean(np.array(scores))
        return average_score, np.array(scores)

    def method(self):
        return "Rouge"
