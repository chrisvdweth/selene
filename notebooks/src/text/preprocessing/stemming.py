import re


class MyPorterStemmer():

    def __init__(self):
        self.vowels = frozenset(["a", "e", "i", "o", "u"])

    def _is_consonant(self, word, i):
        """ """
        if word[i] in self.vowels:
            return False
        if word[i] == "y":
            if i == 0:
                return True
            else:
                return not self._is_consonant(word, i - 1)
        return True

    def _cv_sequence(self, stem):
        return "".join([ "c" if self._is_consonant(stem, i) else "v" for i, l in enumerate(stem)])
    
    def _measure(self, stem):
        """ """
        #
        cv_sequence = self._cv_sequence(stem)
        #
        return cv_sequence.count("vc")
        

    #################################################################################
    # Conditions
    #################################################################################
    
    def _has_positive_measure(self, stem):
        """ """
        return self._measure(stem) > 0
    
    def _contains_vowel(self, stem):
        """ """
        #
        for i in range(len(stem)):
            if not self._is_consonant(stem, i):
                return True
        return False
    
    def _ends_double_consonant(self, word):
        #
        return (
            len(word) >= 2
            and word[-1] == word[-2]
            and self._is_consonant(word, len(word) - 1)
        )
    
    def _ends_cvc(self, word):
        #
        return (
            len(word) >= 3
            and self._is_consonant(word, len(word) - 3)
            and not self._is_consonant(word, len(word) - 2)
            and self._is_consonant(word, len(word) - 1)
            and word[-1] not in ("w", "x", "y")
        )


    #################################################################################
    # Auxiliary methods for performing stemming rules
    #################################################################################    
    
    def _replace_suffix(self, word, suffix, replacement):
        """ """
        #
        assert word.endswith(suffix), "Given word doesn't end with given suffix"
        if suffix == "":
            return word + replacement
        else:
            return word[: -len(suffix)] + replacement

    def _apply_rule_list(self, word, rules):
        """Applies the first applicable suffix-removal rule to the word

        Takes a word and a list of suffix-removal rules represented as
        3-tuples, with the first element being the suffix to remove,
        the second element being the string to replace it with, and the
        final element being the condition for the rule to be applicable,
        or None if the rule is unconditional.
        """
        for rule in rules:
            suffix, replacement, condition = rule
            if suffix == "*d" and self._ends_double_consonant(word):
                stem = word[:-2]
                if condition is None or condition(stem):
                    return stem + replacement
                else:
                    # Don't try any further rules
                    return word
            if word.endswith(suffix):
                stem = self._replace_suffix(word, suffix, "")
                if condition is None or condition(stem):
                    return stem + replacement
                else:
                    # Don't try any further rules
                    #return word
                    pass

        return word

    
    #################################################################################
    # Core steps of Porter Stemmer algorithm
    #################################################################################
    
    def _step1a(self, word):
        rules = [
            ("sses", "ss", None),  # SSES -> SS
            ("ies", "i", None),  # IES  -> I
            ("ss", "ss", None),  # SS   -> SS
            ("s", "", None),  # S    ->
        ]
        return self._apply_rule_list(word, rules)
        

    def _step1b(self, word):
        """ """
        # (m>0) EED -> EE
        if word.endswith("eed"):
            stem = self._replace_suffix(word, "eed", "")
            if self._measure(stem) > 0:
                return stem + "ee"
            else:
                return word

        rule_2_or_3_succeeded = False

        for suffix in ["ed", "ing"]:
            if word.endswith(suffix):
                intermediate_stem = self._replace_suffix(word, suffix, "")
                if self._contains_vowel(intermediate_stem):
                    rule_2_or_3_succeeded = True
                    break

        if not rule_2_or_3_succeeded:
            return word

        rules = [
            ("at", "ate", None),  # AT -> ATE
            ("bl", "ble", None),  # BL -> BLE
            ("iz", "ize", None),  # IZ -> IZE
            ("*d", intermediate_stem[-1], lambda stem: intermediate_stem[-1] not in ("l", "s", "z") ),
            ("", "e", lambda stem: (self._measure(stem) == 1 and self._ends_cvc(stem)) ),
        ]
        return self._apply_rule_list(intermediate_stem, rules)
        

    def _step1c(self, word):
        """ """
        rules =  [
            ("y", "i", (self._contains_vowel) )
        ]
        return self._apply_rule_list(word, rules)
        

    def _step2(self, word):
        """  """
        rules = [
            ("ational", "ate", self._has_positive_measure),
            ("tional", "tion", self._has_positive_measure),
            ("enci", "ence", self._has_positive_measure),
            ("anci", "ance", self._has_positive_measure),
            ("izer", "ize", self._has_positive_measure),
            ("abli", "able", self._has_positive_measure),
            ("alli", "al", self._has_positive_measure),
            ("entli", "ent", self._has_positive_measure),
            ("eli", "e", self._has_positive_measure),
            ("ousli", "ous", self._has_positive_measure),
            ("ization", "ize", self._has_positive_measure),
            ("ation", "ate", self._has_positive_measure),
            ("ator", "ate", self._has_positive_measure),
            ("alism", "al", self._has_positive_measure),
            ("iveness", "ive", self._has_positive_measure),
            ("fulness", "ful", self._has_positive_measure),
            ("ousness", "ous", self._has_positive_measure),
            ("aliti", "al", self._has_positive_measure),
            ("iviti", "ive", self._has_positive_measure),
            ("biliti", "ble", self._has_positive_measure),
        ]
        return self._apply_rule_list(word, rules)

        
    def _step3(self, word):
        """  """
        rules = [
            ("icate", "ic", self._has_positive_measure),
            ("ative", "", self._has_positive_measure),
            ("alize", "al", self._has_positive_measure),
            ("iciti", "ic", self._has_positive_measure),
            ("ical", "ic", self._has_positive_measure),
            ("ful", "", self._has_positive_measure),
            ("ness", "", self._has_positive_measure),
        ]
        return self._apply_rule_list(word, rules)

        
    def _step4(self, word):
        """  """
        measure_gt_1 = lambda stem: self._measure(stem) > 1

        rules = [
            ("al", "", measure_gt_1),
            ("ance", "", measure_gt_1),
            ("ence", "", measure_gt_1),
            ("er", "", measure_gt_1),
            ("ic", "", measure_gt_1),
            ("able", "", measure_gt_1),
            ("ible", "", measure_gt_1),
            ("ant", "", measure_gt_1),
            ("ement", "", measure_gt_1),
            ("ment", "", measure_gt_1),
            ("ent", "", measure_gt_1),
            ("ion", "", lambda stem: self._measure(stem) > 1 and stem[-1] in ("s", "t") ),
            ("ou", "", measure_gt_1),
            ("ism", "", measure_gt_1),
            ("ate", "", measure_gt_1),
            ("iti", "", measure_gt_1),
            ("ous", "", measure_gt_1),
            ("ive", "", measure_gt_1),
            ("ize", "", measure_gt_1),
        ]
        return self._apply_rule_list(word, rules)
        

    def _step5a(self, word):
        """  """
        rules = [
            ("e", "", lambda stem: self._measure(stem) > 1),
            ("e", "", lambda stem: self._measure(stem) == 1 and not self._ends_cvc(stem) )
        ]
        return self._apply_rule_list(word, rules)
        


    def _step5b(self, word):
        """  """
        rules = [
            ("ll", "l", lambda stem: self._measure(word[:-1]) > 1)
        ]
        return self._apply_rule_list(word, rules)


    def stem(self, word, to_lowercase=True):
        #
        stem = word.lower() if to_lowercase else word
        #
        stem = self._step1a(stem)
        stem = self._step1b(stem)
        stem = self._step1c(stem)
        stem = self._step2(stem)
        stem = self._step3(stem)
        stem = self._step4(stem)
        stem = self._step5a(stem)
        stem = self._step5b(stem)
        #
        return stem
