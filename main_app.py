from multiapp import MultiApp
import app, app_about, app_algo

myapp = MultiApp()

# Add all your application here
st.sidebar.title("MENU")
myapp.add_app("Home", app.app)
myapp.add_app("About SER", app_about.app)
myapp.add_app("Algorithms", app_algo.app)

# The main app
myapp.run()