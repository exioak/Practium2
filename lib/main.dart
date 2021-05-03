import 'package:flutter/material.dart';
import 'package:avatar_glow/avatar_glow.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt;
import 'voiceScreen.dart';
import 'textscreen.dart';

const PRIMARY = "primary";
const WHITE = "white";
const BLACK = "black";

const Map<String, Color> myColors = {
  PRIMARY: Color.fromRGBO(220, 220, 220, 1),
  //WHITE: Colors.white,
  WHITE: Color.fromRGBO(0, 0, 0, 1),
  BLACK: Colors.black,
};

class Personal extends StatefulWidget {
  @override
  _PersonalState createState() => _PersonalState();
}

// class _PersonalState extends State<Personal> {
//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       backgroundColor: myColors[PRIMARY],
//       body: _Body(),
//     );
//   }
// }
class _PersonalState extends State<Personal> {
  @override
  Widget build(BuildContext context) {
    return new MaterialApp(
      debugShowCheckedModeBanner: false,
      home: new Scaffold(
        backgroundColor: myColors[PRIMARY],
        body: _Body(),
      ),
    );
  }
}

class _Body extends StatefulWidget {
  @override
  __BodyState createState() => __BodyState();
}

class __BodyState extends State<_Body> {
  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Container(
          margin: EdgeInsets.only(left: 20.0, right: 8.0, top: 40),
          child: Text(
            'Summarizer ',
            style: TextStyle(
              color: Colors.black,
              fontSize: 48,
              fontWeight: FontWeight.w900,
              fontFamily: 'Helvetica',
            ),
          ),
        ),
        SizedBox(
          height: 200,
        ),
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            Expanded(
              child: Container(
                padding: EdgeInsets.fromLTRB(0, 55, 0, 0),
                margin: EdgeInsets.only(left: 20.0, right: 10.0, top: 10.0),
                height: 140.0,
                width: 100.0,
                decoration: BoxDecoration(
                    color: myColors[WHITE],
                    borderRadius: BorderRadius.circular(30)),
                child: Column(children: [
                  GestureDetector(
                      onTap: () => Navigator.push(context,
                          MaterialPageRoute(builder: (_) => VoiceScreen())),
                      child: Container(
                        child: Text(
                          'Voice Input',
                          style: TextStyle(
                            color: Color(0xff57BE70),
                            fontSize: 28,
                            fontWeight: FontWeight.w900,
                            fontFamily: 'Helvetica',
                          ),
                        ),
                      )),
                ]),
              ),
            ),
            Expanded(
              child: Container(
                padding: EdgeInsets.fromLTRB(0, 55, 0, 0),
                margin: EdgeInsets.only(left: 10.0, right: 20.0, top: 10.0),
                height: 140.0,
                width: 100.0,
                //child: Text('Personal'),
                decoration: BoxDecoration(
                    color: myColors[WHITE],
                    borderRadius: BorderRadius.circular(30)),
                child: Column(children: [
                  GestureDetector(
                    child: Container(
                      child: Text(
                        'Text Input',
                        style: TextStyle(
                          color: Color(0xff57BE70),
                          fontSize: 28,
                          fontWeight: FontWeight.w900,
                          fontFamily: 'Helvetica',
                        ),
                      ),
                    ),
                    onTap: () {
                      Navigator.push(
                          context,
                          MaterialPageRoute(
                              builder: (context) => TextScreen()));
                    },
                  ),
                ]),
              ),
            ),
          ],
        ),
      ],
    );
  }
}

void main() {
  runApp(Personal());
}
