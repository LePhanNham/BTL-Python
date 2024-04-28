import numpy as np # sử dụng thư viện numpy cho phép làm việc với mảng và ma trận lớn với tốc độ xử lý nhanh.
import nltk # Dòng này nhập thư viện Natural Language Toolkit (nltk) vào chương trình. Thư viện nltk cung cấp các công cụ và tài liệu để làm việc với ngôn ngữ tự nhiên và xử lý văn bản.
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer # PorterStemmer là một lớp ,trong những thuật toán stemming phổ biến trong xử lý ngôn ngữ tự nhiên.
# Stemming là quá trình giảm các từ về dạng gốc của chúng, được gọi là "stem".bằng cách loại bỏ các tiền tố và hậu tố để tạo ra một dạng chuẩn hóa của từ.
# VD "running" thành "run", "jumps" thành "jump"
# Khởi tạo một đối tượng PorterStemmer
# stemmer = PorterStemmer()
# Thực hiện stemming cho một từ
# word = "jumps"
# stemmed_word = stemmer.stem(word)
# print(stemmed_word)  # Kết quả: "jump"
stemmer = PorterStemmer()

def tokenize(sentence):
    """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """

    # sentence : câu hoặc đoạn văn cần chia thành các từ
    # nltk.word_tokenize : Phương thức trong thư viện Natural Language Toolkit (NLTK) phân tích một câu thành các từ và dấu câu
    # Trả về 1 mảng list chứa các từ
    # text = "Hello, how are you today?"
    # ['Hello', ',', 'how', 'are', 'you', 'today', '?']
    return nltk.word_tokenize(sentence)
def stem(word):
    """
    stemming = find the root form of the word
    examples:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    # Chuyển danh sách các từ thành chữ thường
    # đối tượng stemmer của lớp PorterStemmer() sẽ stem các từ về dạng gốc trong danh sách và trả lại mảng tương ứng vs các kết quả stem
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # words : mảng list chứa danh sách các từ khóa để kiểm tra xem từ đó có xuất hiện trong câu chat hay không
    # sentence : mảng list chứa các từ đã được tách ra từ câu chat

    # B1 : stem các từ trong sentences về dạng nguyên gốc của nó (đã được tokenize)
    # B2 : Khởi tạo mảng bag với toàn bộ giá trị bằng 0 chiều dài bằng words
    # B3 : Duyệt các từ trong words xem nếu có từ nào trong sentences ko thì tại vị trí đó bằng 1
    # B4 : trả lại mảng trùng vị trí

    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag
