import pandas as pd
import re
import emoji
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

class Processor:
    def drop_missing_messsage(self,df):
        df.dropna(subset='Message',inplace=True)
        return df
    # def remove_emojis_from_message(self,df):

    #     df.loc[:,'Message'] = df['Message'].apply(lambda x: emoji.replace_emoji(x, replace='') if isinstance(x, str) else x)
    #     # df['Message'] = df['Message'].apply(lambda x: ' '.join(x.split()) if isinstance(x, str) else x)
    #     # df = df[df['Message'].str.contains('ዋጋ', na=False)] # drop the row if it doesn't contain 'ዋጋ' this removes other posts in the channel that are not intended for trading. eg wishing a happy holiday posts
    #     return df
    
    def clean_message(self,df):
        def clean_txt(text):
            # Replace '\n', '\xa0', and '\ufeff' with a single '\n'
            cleaned_text = re.sub(r'[\n\xa0\ufeff]+', '\n', text)
            
            # Replace multiple occurrences of '\n' with a single '\n'
            cleaned_text = re.sub(r'\n+', '\n', cleaned_text)
            
            return cleaned_text
        
        df.loc[:,'Message'] = df['Message'].apply(lambda x: emoji.replace_emoji(x, replace='') if isinstance(x, str) else x)

        df.loc[:,'Message']=df['Message'].apply(clean_txt)
    
        return df['Message']
    
    def filter_amharic(self,df):
        # Define Amharic character range
        amharic_pattern = re.compile(r'[\u1200-\u137F]')

        def is_majority_amharic(text):
            # Checks if 50% or more of the characters in the message are Amharic.
            total_chars = len(text)
            if total_chars == 0:
                return False
            amharic_chars = len(amharic_pattern.findall(text))
            return (amharic_chars / total_chars) >= 0.5

        # Filter rows where the majority of the message is Amharic
        df_filtered = df[df['Message'].apply(is_majority_amharic)]

        return df_filtered

    def label_message(self,message):
        # Define price and phone number patterns
        price_pattern = re.compile(r'\b(?:ዋጋ[-\s]*)?(\d{3,5})\s?ብር\b|\b(\d{3,5})\s?ብር\b')    
        phone_pattern = re.compile(r'(\+2519\d{8}|\b09\d{2}[- ]?\d{2}[- ]?\d{2}[- ]?\d{2}\b)')
        
        # Split tokens by newline
        tokens = message.split('\n')
        filtered_tokens = list(filter(lambda token: token.strip() != '', tokens))
        labels = []
        loc_flag = False  # To flag location entity start
        product_flag = False  # To flag that product labeling is in progress
        
        
        for i, token in enumerate(filtered_tokens):
            # function to check telegram link
            def check_token(token,regEx):
                # Split the token by spaces
                parts = token.split()
        
                # Iterate through each part and check if it matches the regex pattern
                for part in parts:
                    if re.match(regEx, part):
                        return True  # If any part matches, return True
        
                return False  # If no part matches, return False
            def contains_phone_number(token):
            # Use the search function to check if the pattern is in the token
                if phone_pattern.search(token):
                    return True  # Return True if a match is found
                return False  # Return False if no match is found
            
            def contains_price(token):
            # Use the search function to check if the pattern is in the token
                if price_pattern.search(token):
                    return True  # Return True if a match is found
                return False  # Return False if no match is found
            
            # Check if the token is a beginning token (first token) or contains specific keywords
            if i == 0 and all(city not in token for city in ['አዳማ', 'አዲስ አበባ']) and not contains_phone_number(token):
                labels.append('B-Product')  # Mark the first token or specified keywords as B-Product
                product_flag = True # Set product flag
                continue
                
            if (i==1 and not product_flag and not contains_phone_number(token)) or (product_flag and token.isascii()):
                labels.append('B-Product')
                # product_flag=False
                continue
            # Check for phone number and label as 'O'
            if contains_phone_number(token) :
                labels.append('O')
                product_flag = False  # Reset product flag after encountering these
                continue
            
            # Check for price pattern and label as 'B-PRICE'
            if contains_price(token) or 'ዋጋ' in token:
                labels.append('B-PRICE')
                product_flag = False  # Reset product flag
                loc_flag = False  # Reset location flag
                continue
            if any(keyword in token for keyword in ['አድራሻ', 'አድራሻችን']):
                if  any(keyword in token for keyword in ['አዲስ አበባ','አዳማ','ሞል','ሱ.ቁ']):
                    labels.append('B-LOC')
                    continue
                else:
                    loc_flag=True
                    labels.append('O')
                    continue
            if  any(keyword in token for keyword in ['አዲስ አበባ','አዳማ','ህንፃ','ሞል','ሱ.ቁ']):
                # loc_flag=True
                labels.append('B-LOC')
                continue
            # If token is part of a location entity
            if loc_flag:
                if any(keyword in token for keyword in['ቴሌግራም','የቤተሰባችን አባል','እንልካለን','በነፃ','እናደርሳለን']) or check_token(token,r'https?://t\.me/\w+'):
                    loc_flag = False  # Stop marking as location if we hit Telegram or link
                    labels.append('O')
                    continue
                else:
                    labels.append('B-LOC')  # Mark as B-LOC for location entities
                    continue
            
            # If none of the above, label as 'O'
            labels.append('O')
        
        # Return tokens and corresponding labels
        return list(zip(filtered_tokens, labels))

    def convert_to_conll_format(self,labeled_tokens):
        conll_output = []

        for token, label in labeled_tokens:
            # Split the token into individual words
            words = token.split()

            # Check the label and apply the rules
            if label == 'B-Product':
                # First word gets 'B-Product', rest get 'I-Product'
                conll_output.append(f"{words[0]} B-Product")
                for word in words[1:]:
                    conll_output.append(f"{word} I-Product")
            elif label == 'B-PRICE':
                # First word gets 'B-Product', rest get 'I-Price'
                conll_output.append(f"{words[0]} B-Price")
                for word in words[1:]:
                    conll_output.append(f"{word} I-Price")
            elif label == 'B-LOC':
                # First word gets 'B-Product', rest get 'I-Loc'
                conll_output.append(f"{words[0]} B-LOC")
                for word in words[1:]:
                    conll_output.append(f"{word} I-LOC")
            else:
                # For 'O' or other labels, apply 'O' to all words
                for word in words:
                    conll_output.append(f"{word} O")

        # Print the results
    
        return conll_output


    def label_dataset(self,df):
        labeled_messages = []
        
        # Iterate through each message in the dataset
        for message in df['Message']:
            # Label each message
            labeled_message = self.label_message(message)
            conll_formatted_output=self.convert_to_conll_format(labeled_message)
            labeled_messages.append(conll_formatted_output)
            for line in conll_formatted_output:
                print(line)
                
            print() #an empty new line after every message 
        
        # return labeled_messages

