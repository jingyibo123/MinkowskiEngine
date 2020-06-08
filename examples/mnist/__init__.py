import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
sys.path.append(os.path.dirname(file_dir))
sys.path.append(os.path.dirname(os.path.dirname(file_dir)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(file_dir))))