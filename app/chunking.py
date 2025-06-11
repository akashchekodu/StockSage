def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
    return chunks


def sentence_chunking_overlap(text :str,max_words:int = 120,overlap:int =30) -> List[str]:
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        words = sentence.split()
        if current_len + len(words) > max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:] if overlap else []
            current_len = sum(len(sent.split()) for sent in current_chunk)
        current_chunk.append(sentence)
        current_len += len(words)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks    


def recursive_chunking(text: str, max_words: int = 150, overlap: int = 30) -> List[str]:
    def split_if_too_long(txt: str) -> List[str]:
        words = txt.split()
        if len(words) <= max_words:
            return [txt]
        
        # Try splitting by paragraphs
        paras = [p.strip() for p in txt.split('\n') if p.strip()]
        if len(paras) > 1:
            chunks = []
            for para in paras:
                chunks.extend(split_if_too_long(para))  # recurse
            return chunks
        
        # If still too long, try splitting by sentences
        sentences = [sent.text.strip() for sent in nlp(txt).sents if sent.text.strip()]
        chunks = []
        current_chunk = []
        current_len = 0

        for sentence in sentences:
            words = sentence.split()
            if current_len + len(words) > max_words:
                chunks.append(" ".join(current_chunk))
                current_chunk = current_chunk[-overlap:] if overlap else []
                current_len = sum(len(s.split()) for s in current_chunk)
            current_chunk.append(sentence)
            current_len += len(words)
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    return split_if_too_long(text)
