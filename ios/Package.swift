// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "passive_liveness",
    platforms: [
        .iOS(.v12)
    ],
    products: [
        .library(
            name: "passive_liveness",
            targets: ["passive_liveness"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/readdle/tensorflow-lite-swift.git", from: "2.17.0")
    ],
    targets: [
        .target(
            name: "passive_liveness",
            dependencies: [
                .product(name: "TensorFlowLite", package: "tensorflow-lite-swift")
            ],
            path: "Classes"
        )
    ]
)
