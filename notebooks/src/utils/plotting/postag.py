import matplotlib.pyplot as plt
from wordcloud import WordCloud


def show_wordcloud(source, max_words=50):
    try:
        wordcloud = WordCloud(max_words=1000)
        if type(source).__name__ == 'str' or type(source).__name__ == 'unicode':
            wordcloud.generate_from_text(source)
        else:
            wordcloud.generate_from_frequencies(source)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
    except:
        raise ValueError("Invalid data type for source parameter: str or [(str,float)]")
           
