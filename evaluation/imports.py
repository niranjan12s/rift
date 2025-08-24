import torch
import torch.nn as nn # Import nn
from torch.nn import functional as F # Import F
import os
import nltk
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from collections import Counter
import re # Import regex for cleaning generated text
import math # Import math
