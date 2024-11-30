import {Button} from "@/components/ui/button";
import {ArrowRight, Cpu, LibraryIcon} from "lucide-react";
import React, {useState} from "react";
import CryptoHelper from "@/app/encrypt/helper";
import CodeCard from "@/app/code-card";
import {darcula} from "react-syntax-highlighter/dist/esm/styles/prism";
import Image from 'next/image';


export default function Home() {
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [password, setPassword] = useState("");

    const encryptedUrl = "U2FsdGVkX1+rPasqXOY09ykLCGjIAakK+wwWuQ1GRU/eVmxh1uCA3NtyZ5+UMyLQ0Ean8tW5teuXvxkZSNG4Sg==";

    const goTo = CryptoHelper(encryptedUrl);

    const showPasswordModal = () => {
        setIsModalOpen(true); // Open the custom modal
    };

    const handleSubmit = () => {
        if (password) {
            goTo(password);
            setIsModalOpen(false); // Close the modal after password is entered
        } else {
            alert("Please enter a password");
        }
    };

    return (
        <div className={"flex flex-col flex-grow"}>
            <section className="bg-gradient-to-b from-black via-gray-800 to-black py-20 text-white">
                <div className="w-full mx-auto px-4 text-center">
                    <h2 className="text-4xl font-extrabold mb-4">Empower Your Quantitative Trading</h2>
                    <p className="text-xl mb-8">
                        High-performance libraries and frameworks for quant traders, funds, and low latency
                        application developers
                    </p>
                    <Button size="lg" variant="secondary" onClick={showPasswordModal}>
                        Get Started <ArrowRight className="ml-2"/>
                    </Button>
                </div>

                {isModalOpen && (
                    <div className="fixed inset-0 flex justify-center items-center bg-black bg-opacity-50 z-50">
                        <div className="bg-white p-6 rounded-md shadow-lg w-80">
                            <h2 className="text-lg font-bold mb-4 text-black">Enter Password</h2>
                            <input
                                type="password"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                placeholder="Password"
                                className="w-full p-2 border border-gray-300 rounded-md mb-4 text-black"
                            />
                            <div className="flex justify-between">
                                <Button variant="secondary" onClick={() => setIsModalOpen(false)}>
                                    Cancel
                                </Button>
                                <Button variant="default" onClick={handleSubmit}>
                                    Submit
                                </Button>
                            </div>
                        </div>
                    </div>
                )}
            </section>

            <section className="py-20 bg-gradient-to-b from-black to-gray-900" id="projects">
                <div className="container mx-auto">
                    <h3 className="text-3xl font-bold text-white text-center mb-12">Our Core Frameworks</h3>
                    <div className="flex flex-wrap w-full justify-center">
                        <CodeCard
                            className="mx-3 my-6 w-auto md:w-2/5"
                            icon={<Cpu className="w-10 h-10 text-indigo-600"/>}
                            title="dxcore"
                            description="Core library with CUDA code, OpenMP/I and high frequency trading market making, signal, risk and portfolio strategies, in C++."
                            codeSnippet={`#include <dxcore/market_making.h>
#include <dxcore/signal_processing.h>

__global__ void highFrequencyStrategy(MarketData* data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NUM_SECURITIES) {
        float signal = computeSignal(data[tid]);
        updateQuotes(data[tid], signal);
    }
}`}
                            language="cpp"
                        />
                        <CodeCard
                            className="mx-3 my-6 w-auto md:w-2/5"
                            icon={<LibraryIcon className="w-10 h-10 text-purple-600"/>}
                            link="/dxlib"
                            title="dxlib"
                            description="High-level functionalities, interface for Python for dxcore with methods for manipulating data, networking, storage, caching and ML."
                            codeSnippet={`import dxlib as dx

# Load and preprocess data
data = dx.load_market_data('AAPL', '2023-01-01', '2023-06-30')
features = dx.compute_features(data)

# Train ML model
model = dx.MLModel('RandomForest')
model.train(features, target='returns')

# Make predictions
predictions = model.predict(new_data)`}
                            language="python"
                        />
                        <CodeCard
                            className="mx-3 my-6 w-auto"
                            icon={<Image src={'/divergex.svg'} className="rounded-full bg-white w-10 h-10 text-pink-600"  alt={'divergex'} width={40} height={40}/>}
                            title="dxstudio"
                            description="Native app for studying contracts, analyzing investment opportunities and strategies with API interfaces for calling studio GUI methods from other applications."
                            codeSnippet={`from dxstudio import Studio

# Initialize dxstudio
studio = Studio()

# Load a strategy
strategy = studio.load_strategy('mean_reversion.dxs')

# Backtest the strategy
results = studio.backtest(
    strategy,
    start_date='2023-01-01',
    end_date='2023-06-30',
    capital=1000000
)

# Display results in GUI
studio.display_results(results)`}
                            language="python"
                            style={darcula}
                        />
                    </div>
                </div>
            </section>

            <section className="py-20 bg-gradient-to-b from-gray-900 via-gray-800 to-black text-white"
                     id="docs">
                <div className="container mx-auto px-4 text-center pb-20">
                    <h3 className="text-3xl font-bold mb-8">Ready to Elevate Your Trading?</h3>

                    <p className="mt-4">Check out our documentation on <a href="https://divergex.github.io/dxlib/"
                                                                          className="text-blue-500 hover:text-blue-700 underline">dxlib</a> and <a
                        href="https://divergex.github.io/dxcore/"
                        className="text-blue-500 hover:text-blue-700 underline">dxcore</a>.</p>
                </div>
            </section>
        </div>
    );
}