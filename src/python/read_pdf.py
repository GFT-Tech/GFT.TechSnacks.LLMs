import PyPDF2
import pickle
import os
import time
import tiktoken
import pandas
import openai

def save_list_to_disk(my_list, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(my_list, file)

def load_list_from_disk(file_path):
    with open(file_path, 'rb') as file:
        my_list = pickle.load(file)
    return my_list

def bookmark_dict(bookmark_list):
    result = {}
    for item in bookmark_list:
        if isinstance(item, list):
            # recursive call
            result.update(bookmark_dict(item))
        else:
             result[reader.get_destination_page_number(item)+1] = item.title
    return result

def get_bookmark_list(outline, previous_section=None):
    result = []
    found_section = previous_section
    for item in outline:
        if isinstance(item, list):
            # recursive call
            result += get_bookmark_list(item, found_section)
            #after returned, we set the found_section with the last item on the list
            # so we keep using the right section to set the numbers further
            found_section = result[-1]
        else:
            section_page_number = reader.get_destination_page_number(item)
            if found_section is not None:
                if found_section['start_page']>section_page_number: #is on same page!
                    found_section['end_page']=found_section['start_page']
                else:
                    found_section['end_page']=section_page_number
            found_section = { 
                'title': item.title, 
                'start_page':section_page_number+1
                }
            result += [found_section]
    if previous_section is None:
        found_section['end_page']=len(reader.pages)
    return result


def extract_bookmart_text(reader, bookmarks):    
    for section in bookmarks:
        section_text = ""
        for page_number in range(section['start_page'] - 1, section['end_page'] ):
            page = reader.pages[page_number]
            section_text += page.extract_text()
        section['text'] = section_text

def extract_text_between_pages(reader, start_page, end_page):
    if start_page < 1 or start_page > len(reader.pages):
        raise ValueError("Invalid start page number")

    if end_page < start_page or end_page > len(reader.pages):
        raise ValueError("Invalid end page number")

    text = ""
    for page_number in range(start_page - 1, end_page):
        page = reader.pages[page_number]
        text += page.extract_text()
    return text

# Preparing files
GPT_MODEL = "gpt-3.5-turbo"  # only matters insofar as it selects which tokenizer to use

def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def halved_by_delimiter(string: str, delimiter: str = "\n") -> list[str, str]:
    """Split a string in two, on a delimiter, trying to balance tokens on each side."""
    chunks = string.split(delimiter)
    if len(chunks) == 1:
        return [string, ""]  # no delimiter found
    elif len(chunks) == 2:
        return chunks  # no need to search for halfway point
    else:
        total_tokens = num_tokens(string)
        halfway = total_tokens // 2
        best_diff = halfway
        for i, chunk in enumerate(chunks):
            left = delimiter.join(chunks[: i + 1])
            left_tokens = num_tokens(left)
            diff = abs(halfway - left_tokens)
            if diff >= best_diff:
                break
            else:
                best_diff = diff
        left = delimiter.join(chunks[:i])
        right = delimiter.join(chunks[i:])
        return [left, right]

def truncated_string(
    string: str,
    model: str,
    max_tokens: int,
    print_warning: bool = True,
) -> str:
    """Truncate a string to a maximum number of tokens."""
    encoding = tiktoken.encoding_for_model(model)
    encoded_string = encoding.encode(string)
    truncated_string = encoding.decode(encoded_string[:max_tokens])
    if print_warning and len(encoded_string) > max_tokens:
        print(f"Warning: Truncated string from {len(encoded_string)} tokens to {max_tokens} tokens.")
    return truncated_string


def split_strings_from_subsection(
    subsection,
    max_tokens: int = 1000,
    model: str = GPT_MODEL,
    max_recursion: int = 5,
) -> list[str]:
    """
    Split a subsection into a list of subsections, each with no more than max_tokens.
    Each subsection is a tuple of parent titles [H1, H2, ...] and text (str).
    """
    titles,text = [subsection['title']],subsection['text']
    
    string = "\n\n".join(titles + [text])
    num_tokens_in_string = num_tokens(string)
    # if length is fine, return string
    if num_tokens_in_string <= max_tokens:
        return [string]
    # if recursion hasn't found a split after X iterations, just truncate
    elif max_recursion == 0:
        return [truncated_string(string, model=model, max_tokens=max_tokens)]
    # otherwise, split in half and recurse
    else:
        titles,text = [subsection['title']],subsection['text']
        for delimiter in ["\n\n", "\n", ". "]:
            left, right = halved_by_delimiter(text, delimiter=delimiter)
            if left == "" or right == "":
                # if either half is empty, retry with a more fine-grained delimiter
                continue
            else:
                # recurse on each half
                results = []
                for half in [left, right]:
                    new_half_section = {'title':subsection['title'],'text':half}
                    half_strings = split_strings_from_subsection(
                        new_half_section,
                        max_tokens=max_tokens,
                        model=model,
                        max_recursion=max_recursion - 1,
                    )
                    results.extend(half_strings)
                return results
    # otherwise no split was found, so just truncate (should be very rare)
    return [truncated_string(string, model=model, max_tokens=max_tokens)]

#embedding
def embedd(data):
    # calculate embeddings
    EMBEDDING_MODEL = "gpt-embedding-ada" # this is the deployment on Azure OpenAI
    BATCH_SIZE = 1  # you can submit up to 2048 embedding inputs per request

    #pedro: Adding this to use azure api instead of openai
    openai.api_type = "azure"
    openai.api_base = os.getenv("OPENAI_AZURE_URL") # Azure OpenAI Endpoint
    openai.api_version = "2023-03-15-preview"
    openai.api_key = os.getenv("OPENAI_AZURE_KEY") # Azure OpenAI Key

    embeddings = []
    CT=1
    for batch_start in range(0, len(data), BATCH_SIZE):
        batch_end = batch_start + BATCH_SIZE
        batch = data[batch_start:batch_end]
        print(f"Batch {batch_start} to {batch_end-1}")
        response = openai.Embedding.create(engine=EMBEDDING_MODEL, input=batch)
        for i, be in enumerate(response["data"]):
            assert i == be["index"]  # double check embeddings are in same order as input
        batch_embeddings = [e["embedding"] for e in response["data"]]
        embeddings.extend(batch_embeddings)
        if CT % 9 == 0:  # Check if the number is divisible by 9
            # Perform your operation here
            print("Waiting 1 sec. to avoid rate limit")
            time.sleep(1)  # Introduce a 1-second delay
        CT+=1

    return pandas.DataFrame({"text": data, "embedding": embeddings})

original_pdf='docs/sphinx_open_DesignersGuide.pdf'
processed_file='parsed_pdf.pkl'
embedded_file='parsed_pdf.emb'
if not os.path.exists(embedded_file):
    contents=[]
    if not os.path.exists(processed_file):
        print('Not Found! Processing')
        reader = PyPDF2.PdfReader(original_pdf)
        #bookmarks = bookmark_dict(reader.outline)
        #print(bookmarks)
        #for page,section in bookmarks.items():
        #    print(page,section)
        contents = get_bookmark_list(reader.outline)
        extract_bookmart_text(reader,contents)
        save_list_to_disk(contents, processed_file)
    else:
        print('Loaded from disk!')
        contents = load_list_from_disk(processed_file)

    print(contents[1]['text'])
    # split sections into chunks
    MAX_TOKENS = 1600
    pdf_chunks = []
    for section in contents:
        pdf_chunks.extend(split_strings_from_subsection(section, max_tokens=MAX_TOKENS))

    print(f"{len(contents)} sections split into {len(pdf_chunks)} strings.")
    df=embedd(pdf_chunks)
    df.to_csv(embedded_file, index=False)
else:
    # print example data
    print("File already in place")

# Usage example
#print_outline_titles('docs/sphinx_open_DesignersGuide.pdf')

# Usage example
#read_pdf('docs/sphinx_open_DesignersGuide.pdf')
