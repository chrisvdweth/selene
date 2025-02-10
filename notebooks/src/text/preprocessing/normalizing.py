import re


class EmoticonNormalizer:

    EMOTICON_NORMALIZER__SENTIMENT_UNKNOWN = 10
    EMOTICON_NORMALIZER__SENTIMENT_NEUTRAL = 11
    EMOTICON_NORMALIZER__SENTIMENT_POSITIVE = 12
    EMOTICON_NORMALIZER__SENTIMENT_NEGATIVE = 13
    EMOTICON_NORMALIZER__EMOTICONS_GENERIC_PATTERN_TOP = '[]}{()<>'
    EMOTICON_NORMALIZER__EMOTICONS_GENERIC_PATTERN_EYES = '.:;8BX='
    EMOTICON_NORMALIZER__EMOTICONS_GENERIC_PATTERN_NOSES = '-=~\'^o'
    EMOTICON_NORMALIZER__EMOTICONS_GENERIC_PATTERN_MOUTHS = ')(/\|DPp[]{}<>oO*'
    EMOTICON_NORMALIZER__EMOTICONS_PATTERNS_ENM_MOUTH_POSITIVE = ')]}DPp>*'
    EMOTICON_NORMALIZER__EMOTICONS_PATTERNS_ENM_MOUTH_NEGATIVE = '([{\/<|'
    EMOTICON_NORMALIZER__EMOTICONS_PATTERNS_MNE_MOUTH_POSITIVE = '([{<'
    EMOTICON_NORMALIZER__EMOTICONS_PATTERNS_MNE_MOUTH_NEGATIVE = ')]}\/>|'
    EMOTICON_NORMALIZER__ORIENTATION_UNKNOWN = 100
    EMOTICON_NORMALIZER__ORIENTATION_EYES_NOSE_MOUTH = 101
    EMOTICON_NORMALIZER__ORIENTATION_MOUTH_NOSE_EYES = 102
    EMOTICON_NORMALIZER__SENTIMENT_POSITIVE_STRING = "[EMOTICON+]"
    EMOTICON_NORMALIZER__SENTIMENT_NEUTRAL_STRING = "[EMOTICON0]"
    EMOTICON_NORMALIZER__SENTIMENT_NEGATIVE_STRING = "[EMOTICON-]"
    EMOTICON_NORMALIZER__EMOTICON_MAPPING = { EMOTICON_NORMALIZER__SENTIMENT_POSITIVE : EMOTICON_NORMALIZER__SENTIMENT_POSITIVE_STRING,
                                              EMOTICON_NORMALIZER__SENTIMENT_NEUTRAL : EMOTICON_NORMALIZER__SENTIMENT_NEUTRAL_STRING,
                                              EMOTICON_NORMALIZER__SENTIMENT_NEGATIVE : EMOTICON_NORMALIZER__SENTIMENT_NEGATIVE_STRING }

    
    def __init__(self):
        self.regex_pattern_eyes_nose_mouth = re.compile("([%s]?)([%s])([%s]?)([%s]+)" % 
                                                        tuple(map(re.escape, [EmoticonNormalizer.EMOTICON_NORMALIZER__EMOTICONS_GENERIC_PATTERN_TOP,
                                                                              EmoticonNormalizer.EMOTICON_NORMALIZER__EMOTICONS_GENERIC_PATTERN_EYES,
                                                                              EmoticonNormalizer.EMOTICON_NORMALIZER__EMOTICONS_GENERIC_PATTERN_NOSES,
                                                                              EmoticonNormalizer.EMOTICON_NORMALIZER__EMOTICONS_GENERIC_PATTERN_MOUTHS]
                                                        )))
        self.regex_pattern_mouth_nose_eyes = re.compile("([%s]+)([%s]?)([%s])([%s]?)" % 
                                                        tuple(map(re.escape, [EmoticonNormalizer.EMOTICON_NORMALIZER__EMOTICONS_GENERIC_PATTERN_MOUTHS,
                                                                              EmoticonNormalizer.EMOTICON_NORMALIZER__EMOTICONS_GENERIC_PATTERN_NOSES,
                                                                              EmoticonNormalizer.EMOTICON_NORMALIZER__EMOTICONS_GENERIC_PATTERN_EYES,
                                                                              EmoticonNormalizer.EMOTICON_NORMALIZER__EMOTICONS_GENERIC_PATTERN_TOP]
                                                        )))



    def tag(self, emoticon):
        orientation = self._detect_orientation(emoticon)
        
        if orientation == EmoticonNormalizer.EMOTICON_NORMALIZER__ORIENTATION_UNKNOWN:
            return (EmoticonNormalizer.EMOTICON_NORMALIZER__SENTIMENT_UNKNOWN, emoticon)

        top, eyes, nose, mouth = self._split_emoticon(emoticon, orientation)

        mouth_sentiment, normalized_mouth_size, normalized_mouth = self._analyze_mouth(mouth, orientation)

        if orientation == EmoticonNormalizer.EMOTICON_NORMALIZER__ORIENTATION_EYES_NOSE_MOUTH:
            emoticon_normalized = top + eyes + nose + normalized_mouth
        else:
            emoticon_normalized = normalized_mouth + nose + eyes + top

        return (mouth_sentiment, emoticon_normalized)



    def _split_emoticon(self, emoticon, orientation):
        if orientation == EmoticonNormalizer.EMOTICON_NORMALIZER__ORIENTATION_EYES_NOSE_MOUTH:
            regex_pattern = self.regex_pattern_eyes_nose_mouth
        elif orientation == EmoticonNormalizer.EMOTICON_NORMALIZER__ORIENTATION_MOUTH_NOSE_EYES:
            regex_pattern = self.regex_pattern_mouth_nose_eyes
        else:
            return (None, None, None, None)

        regex_match = regex_pattern.match(emoticon)
        if regex_match is None:
            return None, None, None, None

        emoticon_parts = regex_match.groups()

        if len(emoticon_parts) != 4:
            return None, None, None, None

        if orientation == EmoticonNormalizer.EMOTICON_NORMALIZER__ORIENTATION_EYES_NOSE_MOUTH:
            top = emoticon_parts[0]
            eyes = emoticon_parts[1]
            nose = emoticon_parts[2]
            mouth = emoticon_parts[3]
        elif orientation == EmoticonNormalizer.EMOTICON_NORMALIZER__ORIENTATION_MOUTH_NOSE_EYES:
            top = emoticon_parts[3]
            eyes = emoticon_parts[2]
            nose = emoticon_parts[1]
            mouth = emoticon_parts[0]
        else:
            return (None, None, None, None)

        return (top, eyes, nose, mouth)



    def _detect_orientation(self, emoticon):
        # Generic patterns
        m = self.regex_pattern_eyes_nose_mouth.search(emoticon)
        if m is not None:
            return EmoticonNormalizer.EMOTICON_NORMALIZER__ORIENTATION_EYES_NOSE_MOUTH
        # Generic patterns (mirrored orientation)
        m = self.regex_pattern_mouth_nose_eyes.search(emoticon)
        if m is not None:
            return EmoticonNormalizer.EMOTICON_NORMALIZER__ORIENTATION_MOUTH_NOSE_EYES

        return (EmoticonNormalizer.EMOTICON_NORMALIZER__ORIENTATION_UNKNOWN)



    def _analyze_mouth(self, mouth, orientation):

        if orientation == EmoticonNormalizer.EMOTICON_NORMALIZER__ORIENTATION_EYES_NOSE_MOUTH:
            character_list_positive = EmoticonNormalizer.EMOTICON_NORMALIZER__EMOTICONS_PATTERNS_ENM_MOUTH_POSITIVE
            character_list_negative = EmoticonNormalizer.EMOTICON_NORMALIZER__EMOTICONS_PATTERNS_ENM_MOUTH_NEGATIVE
        elif orientation == EmoticonNormalizer.EMOTICON_NORMALIZER__ORIENTATION_MOUTH_NOSE_EYES:
            character_list_positive = EmoticonNormalizer.EMOTICON_NORMALIZER__EMOTICONS_PATTERNS_MNE_MOUTH_POSITIVE
            character_list_negative = EmoticonNormalizer.EMOTICON_NORMALIZER__EMOTICONS_PATTERNS_MNE_MOUTH_NEGATIVE
        else:
            return (EmoticonNormalizer.EMOTICON_NORMALIZER__SENTIMENT_UNKNOWN, None, None)

        try:
            count_positive = sum(len(c) for c in mouth if c in character_list_positive)
        except:
            count_positive = 0
        try:
            count_negative = sum(len(c) for c in mouth if c in character_list_negative)
        except:
            count_negative = 0

        normalized_mouth_size = int(self._sign(count_positive) + self._sign(count_negative))

        normalized_mouth = mouth[0] * normalized_mouth_size

        if count_positive > count_negative:
            return (EmoticonNormalizer.EMOTICON_NORMALIZER__SENTIMENT_POSITIVE, normalized_mouth_size, normalized_mouth)
        elif count_positive < count_negative:
            return (EmoticonNormalizer.EMOTICON_NORMALIZER__SENTIMENT_NEGATIVE, normalized_mouth_size, normalized_mouth)
        else:
            return (EmoticonNormalizer.EMOTICON_NORMALIZER__SENTIMENT_UNKNOWN, normalized_mouth_size, normalized_mouth)


    def normalize(self, token, emoticon_mapping=EMOTICON_NORMALIZER__EMOTICON_MAPPING):
        # Try to normalize token (mainly remove duplicate mouths) and derive sentiment
        sentiment, token_normalized = self.tag(token)

        is_normalized = False
        if token != token_normalized:
            is_normalized = True

        if sentiment in [EmoticonNormalizer.EMOTICON_NORMALIZER__SENTIMENT_POSITIVE, 
                         EmoticonNormalizer.EMOTICON_NORMALIZER__SENTIMENT_NEUTRAL, 
                         EmoticonNormalizer.EMOTICON_NORMALIZER__SENTIMENT_NEGATIVE]:
            return (True, is_normalized, token_normalized, emoticon_mapping[sentiment])
        else:
            return (False, is_normalized, token_normalized, token)


    def _sign(self, number):
        """Will return 1 for positive,
        -1 for negative, and 0 for 0"""
        try:
            return number/abs(number)
        except ZeroDivisionError:
            return 0
