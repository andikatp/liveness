import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:passive_liveness/passive_liveness.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Passive Liveness Example',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue),
        useMaterial3: true,
      ),
      home: const LivenessHomePage(),
    );
  }
}

class LivenessHomePage extends StatefulWidget {
  const LivenessHomePage({super.key});

  @override
  State<LivenessHomePage> createState() => _LivenessHomePageState();
}

class _LivenessHomePageState extends State<LivenessHomePage> {
  final _detector = PassiveLivenessDetector();
  bool _initialized = false;
  String _status = 'Initializing engine...';

  @override
  void initState() {
    super.initState();
    _initDetector();
  }

  Future<void> _initDetector() async {
    try {
      await _detector.initialize();
      setState(() {
        _initialized = true;
        _status = 'Detector engine ready!';
      });
    } catch (e) {
      setState(() {
        _status = 'Init error: $e';
      });
    }
  }

  Future<void> _testSyntheticLiveness() async {
    if (!_initialized) return;

    // Create a 100x100 synthetic image byte array (dummy RGBA bytes)
    final bytes = Uint8List(100 * 100 * 4);
    for (int i = 0; i < bytes.length; i += 4) {
      bytes[i] = 210;     // R
      bytes[i + 1] = 170; // G
      bytes[i + 2] = 140; // B
      bytes[i + 3] = 255; // A
    }

    try {
      final result = await _detector.detectLivenessFromImageBytes(
        bytes,
        boundingBox: const FaceBoundingBox(x: 20, y: 20, width: 60, height: 60),
      );

      setState(() {
        _status = 'Liveness Result: ${result.status.name.toUpperCase()}\n'
            'Real Score: ${(result.realScore * 100).toStringAsFixed(1)}%\n'
            'Logit Diff: ${result.logitDiff.toStringAsFixed(2)}\n'
            'Inference Time: ${result.inferenceTime.inMilliseconds}ms';
      });
    } catch (e) {
      setState(() {
        _status = 'Detection error: $e';
      });
    }
  }

  @override
  void dispose() {
    _detector.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Passive Liveness Demo'),
      ),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(
                Icons.face,
                size: 80,
                color: Colors.blue,
              ),
              const SizedBox(height: 24),
              Text(
                _status,
                textAlign: TextAlign.center,
                style: const TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                ),
              ),
              const SizedBox(height: 32),
              ElevatedButton.icon(
                onPressed: _initialized ? _testSyntheticLiveness : null,
                icon: const Icon(Icons.play_arrow),
                label: const Text('Run Test Liveness'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
