from PyQt5.QtWidgets import QApplication
import sys
from views.main_view import MainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())