
# BEGIN: user added these matplotlib lines to ensure any plots do not pop-up in their UI
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
plt.ioff()
import os
os.environ['TERM'] = 'dumb'
# END: user added these matplotlib lines to ensure any plots do not pop-up in their UI
# filename: read_react_guidelines.py
# execution: true

with open('/workspace/api_server/README_React_Artifacts.md', 'r') as f:
    content = f.read()
    print(content)