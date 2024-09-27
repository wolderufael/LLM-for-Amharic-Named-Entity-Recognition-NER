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
        price_pattern = re.compile(r'\b\d{3,5}\s?ብር\b')
        price_pattern = re.compile(r'\b\d{3,5}\s?ብር\b')
        phone_pattern = re.compile(r'(\+2519\d{8}\b|\b09\d{8}\b)')
        
        tokens = message.split()
        labels = []
        loc_flag = False  # To flag location entity start
        price_flag = False  # To flag that a price has been encountered
        product_flag = False  # To flag that product labeling is in progress
        
        for i, token in enumerate(tokens):
            # Check for phone number, label as 'O' and stop current labeling
            if phone_pattern.match(token):
                labels.append('O')
                # After encountering a phone number, label everything as 'O'
                continue
            
            # Check for 'አድራሻ' or 'አድራሻችን' to start location entity
            if token in ['አድራሻ', 'አድራሻችን']:
                labels.append('O')  # Mark the token itself as 'O'
                loc_flag = True  # Set flag for location
                price_flag = False  # Reset price flag
                product_flag = False  # Stop product labeling
            # If in location entity, mark tokens as B-LOC
            elif loc_flag:
                if token == 'ቴሌግራም' or re.match(r'https?://t\.me/\w+', token):
                    loc_flag = False  # Stop marking as location if we hit Telegram or link
                    labels.append('O')
                else:
                    labels.append('B-LOC')
                    price_flag = False  # Reset price flag
                    product_flag = False  # Stop product labeling
            # Check for 'ዋጋ' or price pattern
            elif token == 'ዋጋ' or price_pattern.match(token):
                labels.append('B-PRICE' if token == 'ዋጋ' else 'I-PRICE')
                price_flag = True  # Set flag for price entity
                loc_flag = False  # Reset location flag
                product_flag = False  # Stop product labeling
            # If no price or location flag, continue labeling product
            elif not price_flag and not loc_flag:
                if product_flag:
                    labels.append('I-Product')  # Continue labeling as I-Product
                else:
                    labels.append('B-Product')  # Start labeling as B-Product
                    product_flag = True  # Set product flag
            # If price or location has been encountered, label as 'O'
            else:
                labels.append('O')

        # Return tokens and corresponding labels
        return list(zip(tokens, labels))


    def label_dataset(self,df):
        labeled_messages = []
        
        # Iterate through each message in the dataset
        for message in df['Message']:
            # Label each message
            labeled_message = self.label_message(message)
            labeled_messages.append(labeled_message)
        
        return labeled_messages

