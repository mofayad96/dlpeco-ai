from ai.models.context_classifier import ContextClassifier

clf = ContextClassifier()

text = """
تذكير: ديدلاين التقارير الشهرية الخميس الجاي.
للتيم الداخلي بس.
مش للشير برا
"""

print(clf.classify(text))