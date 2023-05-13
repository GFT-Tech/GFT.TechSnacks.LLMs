import PyPDF2
import os
import pandas as pd
import ast
from __init__ import logger
import pickle
import tiktoken
import openai
import time

class pdf_handler:
    def __init__(self, pdf_file_path:str , embedding_model:str, gpt_model_to_count_tokens: str="gpt-3.5-turbo"):
        """
        Creates a PDF Handler

        Args:
            pdf_file_path: PDF File to read
            gpt_model_to_count_tokens: GPT Model to use to count tokens(gpt-3.5-turbo)
            embedding_model: Embedding model
        """
        self.pdf_file_path = pdf_file_path
        file_name = os.path.splitext(os.path.basename(pdf_file_path))[0]
        self.pdf_contents_file_path=file_name+".txt"
        self.pdf_embeddings_file_path=file_name+".csv"
        self.gpt_model_to_count_tokens=gpt_model_to_count_tokens
        self.embedding_model=embedding_model

    def get_embeddings(self) -> pd.DataFrame:
        to_return = pd.DataFrame()
        # Lets check if we calculate the embeddings
        # if we already have a file, we can return it straight away!
        if os.path.exists(self.pdf_embeddings_file_path):
            logger.info(f"### {self.pdf_embeddings_file_path} Found! Loading it...")
            to_return = pd.read_csv(self.pdf_embeddings_file_path)
            to_return['embedding'] = to_return['embedding'].apply(ast.literal_eval)           
        else:
            logger.info(f"### No {self.pdf_embeddings_file_path} Found! Will Build it...")
            # We found no embeddings
            # Lets check if we have already a contents file
            bookmarks=[]
            if not os.path.exists(self.pdf_contents_file_path):
                logger.info(f"### No {self.pdf_contents_file_path} Found! Reading the PDF {self.pdf_file_path}...")
                self.pdf_reader = PyPDF2.PdfReader(self.pdf_file_path)
                logger.info(f"### Extracting ' {self.pdf_file_path}' Bookmarks & Content...")
                bookmarks = self._private_extract_bookmarks(self.pdf_reader.outline)
                logger.info(f"### Loading Bookmarks Contents...")
                self._private_save_list_to_disk(bookmarks, self.pdf_contents_file_path)
            else:
                logger.info(f"### {self.pdf_contents_file_path} Found. Reading that instead of the PDF...")
                bookmarks = self._private_load_list_from_disk(self.pdf_contents_file_path)

            # split sections into chunks
            MAX_TOKENS = 1600
            pdf_chunks = []
            for section in bookmarks:
                pdf_chunks.extend(self._private_split_strings_from_subsection(section, max_tokens=MAX_TOKENS))

            logger.info(f"### {len(bookmarks)} sections split into {len(pdf_chunks)} strings. Embedding...")
            to_return=self._private_calculate_embeddings(pdf_chunks)
            to_return.to_csv(self.pdf_embeddings_file_path, index=False)
        return to_return

    def _private_extract_bookmarks(self, outline, previous_section=None):
        """
        Reads the pdf outline, 

        Args:
            previous_section: (used Internally to track the previous section recursevely)

        Returns:
            Returns a list with items on the following structure:
            {'title':'Section Title', 'start_page':1, 'end_page':2} 
        """
        result = []
        found_section = previous_section
        for item in outline:
            if isinstance(item, list):
                # recursive call
                result += self._private_extract_bookmarks(item, found_section)
                #after returned, we set the found_section with the last item on the list
                # so we keep using the right section to set the numbers further
                found_section = result[-1]
            else:
                section_page_number = self.pdf_reader.get_destination_page_number(item)
                if found_section is not None:
                    if found_section['start_page']>section_page_number: #is on same page!
                        found_section['end_page']=found_section['start_page']
                    else:
                        found_section['end_page']=section_page_number
                    #Now we can get the conteents
                    self._private_extract_bookmark_text(found_section)
                found_section = { 
                    'title': item.title, 
                    'start_page':section_page_number+1
                    }
                result += [found_section]
        if previous_section is None:
            if found_section is None:
                #Absolutelly No Sections were found!
                #We create a single session
                found_section = { 
                    'title': os.path.splitext(os.path.basename(self.pdf_file_path))[0], 
                    'start_page':1                    
                    }
                result += [found_section]
            found_section['end_page']=len(self.pdf_reader.pages)
            self._private_extract_bookmark_text(found_section)
               
        return result
    
    
    def _private_extract_bookmark_text(self, bookmark):
        """
        Extracts the Text of the pages belonging to the section

        Args:
            bookmark: The bookmark, including start_page & end_page

        Returns:
            Will set a 'text' property on the bookmark            
        """        
        section_text = ""
        for page_number in range(bookmark['start_page'] - 1, bookmark['end_page'] ):
            page = self.pdf_reader.pages[page_number]
            section_text += page.extract_text()
        bookmark['text'] = section_text    

    def _private_save_list_to_disk(self,list_to_save, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(list_to_save, file)

    def _private_load_list_from_disk(self,file_path):
        with open(file_path, 'rb') as file:
            loaded_list = pickle.load(file)
        return loaded_list
    

    def _private_split_strings_from_subsection(
        self,
        subsection,
        max_tokens: int = 1000,
        max_recursion: int = 5,
    ) -> list[str]:
        """
        Split a subsection into a list of subsections, each with no more than max_tokens.
        Each subsection is a tuple of parent titles [H1, H2, ...] and text (str).
        """
        titles,text = [subsection['title']],subsection['text']
        
        string = "\n\n".join(titles + [text])
        num_tokens_in_string = self._private_count_tokens(string)
        # if length is fine, return string
        if num_tokens_in_string <= max_tokens:
            return [string]
        # if recursion hasn't found a split after X iterations, just truncate
        elif max_recursion == 0:
            return [self._private_truncated_string(string, max_tokens=max_tokens)]
        # otherwise, split in half and recurse
        else:
            titles,text = [subsection['title']],subsection['text']
            for delimiter in ["\n\n", "\n", ". "]:
                left, right = self._private_halved_by_delimiter(text, delimiter=delimiter)
                if left == "" or right == "":
                    # if either half is empty, retry with a more fine-grained delimiter
                    continue
                else:
                    # recurse on each half
                    results = []
                    for half in [left, right]:
                        new_half_section = {'title':subsection['title'],'text':half}
                        half_strings = self._private_split_strings_from_subsection(
                            new_half_section,
                            max_tokens=max_tokens,
                            max_recursion=max_recursion - 1,
                        )
                        results.extend(half_strings)
                    return results
        # otherwise no split was found, so just truncate (should be very rare)
        return [self._private_truncated_string(string, max_tokens=max_tokens)]

    def _private_count_tokens(self,text: str) -> int:
        """Return the number of tokens in a string."""
        encoding = tiktoken.encoding_for_model(self.gpt_model_to_count_tokens)
        return len(encoding.encode(text))
    
    def _private_truncated_string(
        self,
        string: str,
        max_tokens: int,
        print_warning: bool = True,
    ) -> str:
        """Truncate a string to a maximum number of tokens."""
        encoding = tiktoken.encoding_for_model(self.gpt_model_to_count_tokens)
        encoded_string = encoding.encode(string)
        truncated_string = encoding.decode(encoded_string[:max_tokens])
        if print_warning and len(encoded_string) > max_tokens:
            logger.warning(f"Truncated string from {len(encoded_string)} tokens to {max_tokens} tokens.")
        return truncated_string

    def _private_halved_by_delimiter(self, string: str, delimiter: str = "\n") -> list[str, str]:
        """Split a string in two, on a delimiter, trying to balance tokens on each side."""
        chunks = string.split(delimiter)
        if len(chunks) == 1:
            return [string, ""]  # no delimiter found
        elif len(chunks) == 2:
            return chunks  # no need to search for halfway point
        else:
            total_tokens = self._private_count_tokens(string)
            halfway = total_tokens // 2
            best_diff = halfway
            for i, chunk in enumerate(chunks):
                left = delimiter.join(chunks[: i + 1])
                left_tokens = self._private_count_tokens(left)
                diff = abs(halfway - left_tokens)
                if diff >= best_diff:
                    break
                else:
                    best_diff = diff
            left = delimiter.join(chunks[:i])
            right = delimiter.join(chunks[i:])
            return [left, right]
        
    def _private_calculate_embeddings(self,data):
        # calculate embeddings
        # OpenAI you can submit up to 2048 embedding inputs per request
        # On Azure we can only 1! :-(
        BATCH_SIZE = 1  

        embeddings = []
        if openai.api_type=="azure":
            CT=1
        else:
            CT=1000
        for batch_start in range(0, len(data), BATCH_SIZE):
            batch_end = batch_start + BATCH_SIZE
            batch = data[batch_start:batch_end]
            logger.info(f"### Batch {batch_start} to {batch_end-1}")
            response = openai.Embedding.create(engine=self.embedding_model, input=batch)
            for i, be in enumerate(response["data"]):
                assert i == be["index"]  # double check embeddings are in same order as input
            batch_embeddings = [e["embedding"] for e in response["data"]]
            embeddings.extend(batch_embeddings)
             # If we make 10 calls withion a second we get over the limit of Azure...

            if openai.api_type=="azure" and CT % 9 == 0: 
                # Perform your operation here
                logger.info(f"### Waiting 1 sec. to avoid rate limit")
                time.sleep(1)  # Introduce a 1-second delay
            CT+=1

        return pd.DataFrame({"text": data, "embedding": embeddings})        