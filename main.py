import re
import numpy as np

np.set_printoptions(precision=3)

def parse_document(doc: str) -> list[str]:
    return re.sub(r'[^\w\s]', '', doc).lower().split(' ')

def get_unique_words(*lists):
    words = set()

    for arr in lists:
        for word in arr:
            words.add(word)

    return words


def compute_c_q(parsed_docs: list[list[str]], query: list[str]) -> tuple[np.array, np.array]:
    all_words = get_unique_words(*parsed_docs, query)

    C = np.zeros((len(all_words), len(parsed_docs)))
    q = np.zeros(len(all_words))

    for i, word in enumerate(all_words):
        if word in query:
            q[i] = 1
        for j, doc in enumerate(parsed_docs):
            if word in doc:
                C[i, j] = 1

        #print(f"{word} | {C[i]} | {q[i]}")

    return C, q

def get_input() -> tuple[np.array, np.array, int]:
    # return C matrix and q vector
    parsed_docs = []
    num_of_documents = int(input())
    for _ in range(num_of_documents):
        parsed_docs.append(parse_document(input()))
        
    query = parse_document(input())
    
    num_of_reduction = int(input())
    
    C, q = compute_c_q(parsed_docs, query)
    
    return C, q, num_of_reduction


def compute_cosine(C_k: np.array, q_k: np.array) -> list:
    output = []

    # transpose, because docs are columns
    for doc in C_k.T:
        relevance = np.dot(doc, q_k) / (np.linalg.norm(doc) * np.linalg.norm(q_k))
        output.append(relevance)

    return output


def compute_relevancy(C: np.array, q: np.array, k: int) -> np.array:
    U, s, Vt = np.linalg.svd(C)

    sk = np.take(s, range(k), axis=0)
    Sk = np.diag(sk)
    Vkt = np.take(Vt, range(k), axis=0)

    Ck = Sk.dot(Vkt)

    Ukt = np.take(U.T, range(k), axis=0)
    Sk1 = np.linalg.inv(Sk)

    q_k = Sk1.dot(Ukt).dot(q)
    
    return compute_cosine(Ck, q_k)


def main():
    C, q, k = get_input()
    relevancy = compute_relevancy(C, q, k)
    formatted_relevancy = list(map(lambda x: round(float(x), 2), relevancy))
    print(formatted_relevancy)
    
    
def test():
    C = np.array([[1,0,1,0,0,0], [0,1,0,0,0,0], [1,1,0,0,0,0], [1,0,0,1,1,0], [0,0,0,1,0,1]])
    q = np.array([1,1,0,0,0])
    k = 3
    
    print(compute_relevancy(C, q, k))


if __name__ == '__main__':
    main()
    