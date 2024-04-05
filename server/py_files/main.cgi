import cgitb
import cgi

cgitb.enable()    
form = cgi.FieldStorage()
#!/usr/bin/env python

print("Content-type: text/html")
print()

print("""
<html>
<head>
<title>CGI Example</title>
</head>
<body>
<h1>Hello Universe </h1>
</body>
</html>
""")