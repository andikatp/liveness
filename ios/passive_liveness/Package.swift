// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "passive_liveness",
    platforms: [
        .iOS("12.0")
    ], 
    products: [
        .library(name: "passive-liveness", targets: ["passive_liveness"])
    ],
    dependencies: [
        .package(name: "FlutterFramework", path: "../FlutterFramework"),
        .package(url: "https://github.com/readdle/tensorflow-lite-swift.git", from: "2.17.0")
    ],
    targets: [
        .target(
            name: "passive_liveness",
            dependencies: [
                .product(name: "FlutterFramework", package: "FlutterFramework"),
                .product(name: "TensorFlowLite", package: "tensorflow-lite-swift")
            ]
        )
    ]
)
