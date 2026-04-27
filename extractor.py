import pandas as pd
import numpy as np
import re
import math
import tldextract
from urllib.parse import urlparse
from collections import Counter

class URLExtractor:
    def __init__(self, url):
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        self.url = url
        self.parsed_url = urlparse(url)
        self.tld_info = tldextract.extract(url)
        
        # Core Components
        self.domain = self.tld_info.registered_domain or ""
        self.subdomain = self.tld_info.subdomain or ""
        self.full_domain = (self.subdomain + '.' + self.domain) if self.subdomain else self.domain
        self.path = self.parsed_url.path or ""
        self.query = self.parsed_url.query or ""
        
    def get_entropy(self, text):
        if not text or len(text) <= 1: return 0.0
        probs = [n/len(text) for n in Counter(text).values()]
        entropy = -sum(p * math.log2(p) for p in probs)
        return entropy / math.log2(len(text))

    def count_ldl(self, text):
        if not text: return 0
        return len(re.findall(r'[a-zA-Z][0-9][a-zA-Z]', text))

    def count_dld(self, text):
        if not text: return 0
        return len(re.findall(r'[0-9][a-zA-Z][0-9]', text))

    def get_longest_word_len(self, text):
        if not text: return 0
        words = re.split(r'\W+', text)
        return max((len(w) for w in words if w), default=0)

    def get_character_continuity_rate(self, text):
        if not text: return 0
        if len(text) == 1: return 1.0
        changes = 0
        for i in range(len(text) - 1):
            if text[i].isalpha() != text[i+1].isalpha() or text[i].isdigit() != text[i+1].isdigit():
                changes += 1
        return (len(text) - changes) / len(text)

    def count_symbols(self, text):
        if not text: return 0
        return sum(not c.isalnum() for c in text)

    def count_delimiters(self, text):
        if not text: return 0
        # Delimiters in this dataset seem to be '.', '-', '_'
        return sum(c in '.-_' for c in text)

    def extract(self):        
        # Tokens
        domain_tokens = [t for t in self.full_domain.split('.') if t]
        path_tokens = [t for t in self.path.split('/') if t]
        
        # Features 
        query_len = len(self.query)
        dom_token_cnt = len(domain_tokens)
        path_token_cnt = len(path_tokens)
        avg_dom_token_len = sum(len(t) for t in domain_tokens)/dom_token_cnt if dom_token_cnt > 0 else 0
        long_dom_token_len = max(len(t) for t in domain_tokens) if domain_tokens else 0
        avg_path_token_len = sum(len(t) for t in path_tokens)/path_token_cnt if path_token_cnt > 0 else 0
        
        tld_val = len(self.tld_info.suffix)
        vowels = len(re.findall(r'[aeiou]', self.url, re.I))
        ace_count = len(re.findall(r'[ace]', self.url, re.I))
        
        # LDL/DLD
        ldl_url = self.count_ldl(self.url)
        ldl_domain = self.count_ldl(self.full_domain)
        ldl_path = self.count_ldl(self.path)
        ldl_filename = self.count_ldl(path_tokens[-1]) if path_tokens else 0
        ldl_getArg = self.count_ldl(self.query)
        
        dld_url = self.count_dld(self.url)
        # Note: dld_domain was dropped from training, so we don't calculate it here.
        dld_path = self.count_dld(self.path)
        dld_filename = self.count_dld(path_tokens[-1]) if path_tokens else 0
        dld_getArg = self.count_dld(self.query)
        
        url_len = len(self.url)
        dom_len = len(self.full_domain)
        path_len = len(self.path)
        subdir_len = len('/'.join(path_tokens[:-1])) if len(path_tokens) > 1 else 0
        file_name_len = len(path_tokens[-1]) if path_tokens else 0
        ext_len = len(self.path.split('.')[-1]) if '.' in self.path else 0
        arg_len = len(self.query)
        
        # Ratios
        path_url_ratio = path_len / url_len if url_len > 0 else 0
        arg_url_ratio = arg_len / url_len if url_len > 0 else 0
        arg_dom_ratio = arg_len / dom_len if dom_len > 0 else 0
        dom_url_ratio = dom_len / url_len if url_len > 0 else 0
        path_dom_ratio = path_len / dom_len if dom_len > 0 else 0
        arg_path_ratio = arg_len / path_len if path_len > 0 else 0
        
        executable = 1 if any(self.url.lower().endswith(ext) for ext in ['.exe', '.bin', '.sh', '.bat', '.apk']) else 0
        dots_count = self.url.count('.')
        
        char_cont_rate = self.get_character_continuity_rate(self.url)
        long_var_val = self.get_longest_word_len(self.query)        
        # Digits/Letters
        url_digits = sum(c.isdigit() for c in self.url)
        host_digits = sum(c.isdigit() for c in self.full_domain)
        dir_digits = sum(c.isdigit() for c in '/'.join(path_tokens[:-1])) if len(path_tokens) > 1 else 0
        file_digits = sum(c.isdigit() for c in path_tokens[-1]) if path_tokens else 0
        ext_digits = sum(c.isdigit() for c in self.path.split('.')[-1]) if '.' in self.path else 0
        query_digits = sum(c.isdigit() for c in self.query)
        
        url_letters = sum(c.isalpha() for c in self.url)
        host_letters = sum(c.isalpha() for c in self.full_domain)
        dir_letters = sum(c.isalpha() for c in '/'.join(path_tokens[:-1])) if len(path_tokens) > 1 else 0
        file_letters = sum(c.isalpha() for c in path_tokens[-1]) if path_tokens else 0
        ext_letters = sum(c.isalpha() for c in self.path.split('.')[-1]) if '.' in self.path else 0
        query_letters = sum(c.isalpha() for c in self.query)
        
        long_path_token = max(len(t) for t in path_tokens) if path_tokens else 0
        sens_word = 1 if any(w in self.url.lower() for w in ['secure', 'login', 'bank', 'update', 'signin']) else 0
        
        # spcharUrl in ISCX-2016 counts hyphens '-' and underscores '_' in the URL
        sp_char_url = self.url.count('-') + self.url.count('_')
        
        has_path = len(path_tokens) > 0
        has_subdir = len(path_tokens) > 1
        has_query = len(self.query) > 0
        has_ext = '.' in self.path

        # Map features to their explicit column names (matching All.csv original columns)
        feature_dict = {
            'Querylength': query_len,
            'domain_token_count': dom_token_cnt,
            'path_token_count': path_token_cnt,
            'avgdomaintokenlen': avg_dom_token_len,
            'longdomaintokenlen': long_dom_token_len,
            'avgpathtokenlen': avg_path_token_len,
            'tld': tld_val,
            'charcompvowels': vowels,
            'charcompace': ace_count,
            'ldl_url': ldl_url,
            'ldl_domain': ldl_domain,
            'ldl_path': ldl_path,
            'ldl_filename': ldl_filename if has_path else -1,
            'ldl_getArg': ldl_getArg if has_query else -1,
            'dld_url': dld_url,
            'dld_domain': -1, # Was removed but we keep structure
            'dld_path': dld_path,
            'dld_filename': dld_filename if has_path else -1,
            'dld_getArg': dld_getArg if has_query else -1,
            'urlLen': url_len,
            'domainlength': dom_len,
            'pathLength': path_len,
            'subDirLen': subdir_len,
            'fileNameLen': file_name_len if has_path else -1,
            'this.fileExtLen': ext_len if has_ext else -1,
            'ArgLen': arg_len,
            'pathurlRatio': path_url_ratio,
            'ArgUrlRatio': arg_url_ratio,
            'argDomanRatio': arg_dom_ratio,
            'domainUrlRatio': dom_url_ratio,
            'pathDomainRatio': path_dom_ratio,
            'argPathRatio': arg_path_ratio,
            'executable': executable,
            'isPortEighty': -1,
            'NumberofDotsinURL': dots_count,
            'ISIpAddressInDomainName': -1,
            'CharacterContinuityRate': char_cont_rate,
            'LongestVariableValue': long_var_val if has_query else -1,
            'URL_DigitCount': url_digits,
            'host_DigitCount': host_digits,
            'Directory_DigitCount': dir_digits if has_subdir else -1,
            'File_name_DigitCount': file_digits if has_path else -1,
            'Extension_DigitCount': ext_digits if has_ext else -1,
            'Query_DigitCount': query_digits if has_query else -1,
            'URL_Letter_Count': url_letters,
            'host_letter_count': host_letters,
            'Directory_LetterCount': dir_letters if has_subdir else -1,
            'Filename_LetterCount': file_letters if has_path else -1,
            'Extension_LetterCount': ext_letters if has_ext else -1,
            'Query_LetterCount': query_letters if has_query else -1,
            'LongestPathTokenLength': long_path_token,
            'Domain_LongestWordLength': self.get_longest_word_len(self.full_domain),
            'Path_LongestWordLength': self.get_longest_word_len(self.path),
            'sub-Directory_LongestWordLength': self.get_longest_word_len('/'.join(path_tokens[:-1])) if has_subdir else -1,
            'Arguments_LongestWordLength': self.get_longest_word_len(self.query) if has_query else -1,
            'URL_sensitiveWord': sens_word,
            'URLQueries_variable': self.query.count('&') + 1 if has_query else -1,
            'spcharUrl': sp_char_url,
            'delimeter_Domain': self.count_delimiters(self.full_domain),
            'delimeter_path': self.count_delimiters(self.path),
            'delimeter_Count': self.count_delimiters(self.query) if has_query else -1,
            'NumberRate_URL': url_digits/url_len if url_len > 0 else 0,
            'NumberRate_Domain': host_digits/dom_len if dom_len > 0 else 0,
            'NumberRate_DirectoryName': dir_digits/subdir_len if subdir_len > 0 else -1,
            'NumberRate_FileName': file_digits/file_name_len if file_name_len > 0 else -1,
            'NumberRate_Extension': ext_digits/ext_len if ext_len > 0 else -1,
            'NumberRate_AfterPath': query_digits/arg_len if arg_len > 0 else -1,
            'SymbolCount_URL': self.count_symbols(self.url),
            'SymbolCount_Domain': self.count_symbols(self.full_domain),
            'SymbolCount_Directoryname': self.count_symbols('/'.join(path_tokens[:-1])) if has_subdir else -1,
            'SymbolCount_FileName': self.count_symbols(path_tokens[-1]) if has_path else -1,
            'SymbolCount_Extension': self.count_symbols(self.path.split('.')[-1]) if has_ext else -1,
            'SymbolCount_Afterpath': self.count_symbols(self.query) if has_query else -1,
            'Entropy_URL': self.get_entropy(self.url),
            'Entropy_Domain': self.get_entropy(self.full_domain),
            'Entropy_DirectoryName': self.get_entropy(self.path) if has_path else -1,
            'Entropy_Filename': self.get_entropy(path_tokens[-1]) if has_path else -1,
            'Entropy_Extension': self.get_entropy(self.path.split('.')[-1]) if has_ext else -1,
            'Entropy_Afterpath': self.get_entropy(self.query) if has_query else -1
        }
        
        return pd.DataFrame([feature_dict])