import pandas as pd
import re
import emoji
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

class Processor:
    def drop_missing_messsage(self,df):
        df.dropna(subset='Message',inplace=True)
        return df
    def remove_emojis_from_message(self,df):

        df.loc[:,'Message'] = df['Message'].apply(lambda x: emoji.replace_emoji(x, replace='') if isinstance(x, str) else x)
        return df
    
    def filter_amharic(df):
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