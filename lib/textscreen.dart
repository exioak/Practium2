import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

class TextScreen extends StatefulWidget {
  @override
  _TextScreenState createState() => _TextScreenState();
}

class _TextScreenState extends State<TextScreen> {
  TextEditingController _text = TextEditingController();
  int number = 5;
  String output = "";

  apiCall() async {
    var res = await http.post(
      Uri.parse("http://9a6b8bea4168.ngrok.io/${_text.text}"),
    );
    if (res.statusCode == 200) {
      // String summ = res.body.;
      setState(() {
        output = res.body;
      });
    } else {
      setState(() {
        output = "Error in API";
      });
    }
  }

  textField() {
    return Container(
      margin: EdgeInsets.all(15.0),
      height: 61,
      child: Row(
        children: [
          Expanded(
            child: Container(
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(35.0),
                boxShadow: [
                  BoxShadow(
                      offset: Offset(0, 3), blurRadius: 5, color: Colors.grey)
                ],
              ),
              child: Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: _text,
                      decoration: InputDecoration(
                          contentPadding: EdgeInsets.all(8),
                          hintText: "Type Something...",
                          border: InputBorder.none),
                    ),
                  ),
                ],
              ),
            ),
          ),
          SizedBox(width: 15),
          Container(
            padding: const EdgeInsets.all(15.0),
            decoration:
                BoxDecoration(color: Colors.black, shape: BoxShape.circle),
            child: InkWell(
              onTap: () {
                apiCall();
              },
              child: Icon(
                Icons.mode_edit,
                color: Colors.white,
              ),
            ),
          )
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Column(
        children: [
          Expanded(
              child: Center(
                  child: Container(
            child: Text(output),
          ))),
          textField(),
        ],
      ),
    );
  }
}
