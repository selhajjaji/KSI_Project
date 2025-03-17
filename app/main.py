# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 14:07:05 2025

@author: m
"""

from app import create_app
from app.routes import configure_routes

app = create_app()
configure_routes(app)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)