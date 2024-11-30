import Link from 'next/link'
import { ArrowRight, Cpu, Database} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import {ReactNode} from "react";

export default function HomePage() {
    return (
        <div className="flex flex-col min-h-screen">
            <header className="bg-background border-b">
                <div className="container mx-auto px-4 py-6 flex justify-between items-center">
                    <h1 className="text-2xl font-bold">divergex</h1>
                    <nav>
                        <ul className="flex space-x-4">
                            <li><Link href="#" className="text-muted-foreground hover:text-primary">Products</Link></li>
                            <li><Link href="#" className="text-muted-foreground hover:text-primary">Documentation</Link></li>
                            <li><Link href="#" className="text-muted-foreground hover:text-primary">Pricing</Link></li>
                        </ul>
                    </nav>
                </div>
            </header>

            <main className="flex-grow">
                <section className="bg-gradient-to-br from-gray-600 via-yellow-800 to-black py-20 text-white">
                    <div className="container mx-auto px-4 text-center">
                        <h2 className="text-4xl font-extrabold mb-4">Empower Your Quantitative Trading</h2>
                        <p className="text-xl mb-8">
                            High-performance libraries and frameworks for quant traders, funds, and low latency application developers
                        </p>
                        <Button size="lg" variant="secondary">
                            Get Started <ArrowRight className="ml-2" />
                        </Button>
                    </div>
                </section>

                <section className="py-20 bg-gradient-to-r from-gray-100 to-gray-200">
                    <div className="container mx-auto px-4">
                        <h3 className="text-3xl font-bold text-center mb-12">Our Core Frameworks</h3>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                            <FrameworkCard
                                icon={<Cpu className="w-10 h-10 text-indigo-600" />}
                                title="dxcore"
                                description="Core library with CUDA code, OpenMP/I and high frequency trading market making, signal, risk and portfolio strategies, in C++."
                                codeSnippet={`
#include <dxcore/market_making.h>
#include <dxcore/signal_processing.h>

__global__ void highFrequencyStrategy(MarketData* data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < NUM_SECURITIES) {
        float signal = computeSignal(data[tid]);
        updateQuotes(data[tid], signal);
    }
}
                `}
                            />
                            <FrameworkCard
                                icon={<Database className="w-10 h-10 text-purple-600" />}
                                title="dxlib"
                                description="High-level functionalities, interface for Python for dxcore with methods for manipulating data, networking, storage, caching and ML."
                                codeSnippet={`
import dxlib as dx

# Load and preprocess data
data = dx.load_market_data('AAPL', '2023-01-01', '2023-06-30')
features = dx.compute_features(data)

# Train ML model
model = dx.MLModel('RandomForest')
model.train(features, target='returns')

# Make predictions
predictions = model.predict(new_data)
                `}
                            />
                            <FrameworkCard
                                icon={<Cpu className="w-10 h-10 text-pink-600" />}
                                title="dxstudio"
                                description="Native app for studying contracts, analyzing investment opportunities and strategies with API interfaces for calling studio GUI methods from other applications."
                                codeSnippet={`
from dxstudio import Studio

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
studio.display_results(results)
                `}
                            />
                        </div>
                    </div>
                </section>

                <section className="py-20 bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white">
                    <div className="container mx-auto px-4 text-center">
                        <h3 className="text-3xl font-bold mb-8">Ready to Elevate Your Trading?</h3>
                        <Button size="lg" variant="secondary">
                            See the Get Started <ArrowRight className="ml-2" />
                        </Button>
                    </div>
                </section>
            </main>

            <footer className="bg-background border-t py-8">
                <div className="container mx-auto px-4 text-center text-muted-foreground">
                    <p>&copy; 2025 divergex. All rights reserved.</p>
                </div>
            </footer>
        </div>
    )
}

interface FrameworkCardProps {
    icon: ReactNode;        // Type for React elements like icons
    title: string;          // Title is a string
    description: string;    // Description is a string
    codeSnippet: string;    // Code snippet is a string
}

function FrameworkCard({ icon, title, description, codeSnippet }: FrameworkCardProps) {
    return (
        <Card className="overflow-hidden">
            <CardHeader>
                <div className="mb-4">{icon}</div>
                <CardTitle>{title}</CardTitle>
            </CardHeader>
            <CardContent>
                <CardDescription className="mb-4">{description}</CardDescription>
                <div className="bg-gray-900 text-gray-100 p-4 rounded-md overflow-x-auto">
          <pre className="text-sm">
            <code>{codeSnippet}</code>
          </pre>
                </div>
            </CardContent>
        </Card>
    );
}