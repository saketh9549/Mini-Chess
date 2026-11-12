"""
Forwarding wrapper so running `python app.py` launches the upgraded `app/gui.py`.
"""
from app.gui import main

if __name__ == "__main__":
    main()

