import 'package:flutter/material.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt;
import 'package:http/http.dart' as http;

class VoiceScreen extends StatefulWidget {
  @override
  _VoiceScreenState createState() => _VoiceScreenState();
}

class _VoiceScreenState extends State<VoiceScreen> {
  String output = "";

  int number = 5;

  stt.SpeechToText _speech;
  bool _isListening = false;
  String _text = 'Press the button and start speaking';
  double _confidence = 1.0;

  initState() {
    super.initState();
    _speech = stt.SpeechToText();
  }

  apiCallVoice() async {
    print(_text);
    var res = await http.post(
      Uri.parse("https://9a6b8bea4168.ngrok.io/${_text}"),
    );
    if (res.statusCode == 200) {
      setState(() {
        output = res.body;
      });
    } else {
      setState(() {
        output = "Error in API";
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Expanded(
              child: Center(
                  child: Container(
            child: Text(output),
          ))),
          Row(children: [
            Container(
              padding: const EdgeInsets.all(15), //all(15.0),
              decoration:
                  BoxDecoration(color: Colors.black, shape: BoxShape.circle),
              child: InkWell(
                onTap: () => _listen(),
                child: Icon(
                  Icons.mic,
                  color: Colors.white,
                ),
              ),
            ),
            SizedBox(width: 25),
            Container(
              padding: const EdgeInsets.all(15.0),
              decoration:
                  BoxDecoration(color: Colors.black, shape: BoxShape.circle),
              child: InkWell(
                onTap: () {
                  apiCallVoice();
                  setState(() {
                    _isListening = false;
                  });
                },
                child: Icon(
                  Icons.done,
                  color: Colors.white,
                ),
              ),
            )
          ])
        ],
      ),
    );
  }

  void _listen() async {
    if (!_isListening) {
      bool available = await _speech.initialize(
        onStatus: (val) => print('onStatus: $val'),
        onError: (val) => print('onError: $val'),
      );
      if (available) {
        print("here");
        setState(() => _isListening = true);
        _speech.listen(
          onResult: (val) => setState(() {
            _text = val.recognizedWords;
            if (val.hasConfidenceRating && val.confidence > 0) {
              _confidence = val.confidence;
            }
          }),
        );
      }
    } else {
      setState(() => _isListening = false);
      _speech.stop();
    }
  }
}
