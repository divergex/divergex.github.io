"use client";
import {Card, CardContent, CardDescription, CardHeader, CardTitle} from "@/components/ui/card";
import React from "react";
import {Button} from "@/components/ui/button";

export default function AboutPage() {
    return (
        <div className="flex flex-col flex-grow py-12 px-6 bg-gradient-to-b from-black via-gray-800 to-black text-white min-h-full">
            <div className="my-5 text-center">
                <h1 className="text-4xl font-bold mb-6">
                    Welcome to Divergex
                </h1>
                <div className={"my-10 mx-auto flex flex-col md:flex-row justify-center md:space-x-10"}>
                    <p className="w-full md:w-1/3 text-lg mb-8 ">
                        Divergex is a FOSS initiative for quantitative and algorithmic trading.
                        We provide a platform for developers to build and deploy trading algorithms.
                        Our platform is built on top of the DX Framework,
                        which is a collection of libraries and tools for quantitative trading.
                    </p>
                    <p className="w-full md:w-1/3 text-lg mb-8">
                        We also provide a marketplace for developers to sell their trading algorithms.
                        Our goal is to democratize algorithmic trading and make it accessible to everyone.
                    </p>
                </div>
            </div>
            <div className={"mb-20 w-1/2 mx-auto h-0.5 bg-white rounded"}/>
            <div className={"grid grid-cols-1 md:grid-cols-3 gap-6"}>
                <div>
                    <Card className="bg-transparent border-white border-2 text-white shadow-lg rounded-lg">
                        <CardHeader>
                            <CardTitle className="text-2xl font-semibold">Our Mission</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <CardDescription>
                                We aim to empower developers and traders by providing an open-source platform and
                                marketplace. Our mission is to remove the barriers to entry in algorithmic trading,
                                making it easier for anyone to participate in the market.
                            </CardDescription>
                        </CardContent>
                    </Card>
                </div>
                <div>
                    <Card className="bg-transparent border-white border-2 text-white shadow-lg rounded-lg">
                        <CardHeader>
                            <CardTitle className="text-2xl font-semibold">The DX Framework</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <CardDescription>
                                The DX Framework is a powerful set of libraries and tools that simplify the development
                                and deployment of trading algorithms. It&#39;s designed for ease of use and maximum
                                performance, giving you the flexibility to innovate.
                            </CardDescription>
                        </CardContent>
                    </Card>
                </div>
                <div>
                    <Card className="bg-transparent border-white border-2 text-white shadow-lg rounded-lg">
                        <CardHeader>
                            <CardTitle className="text-2xl font-semibold">Marketplace for Algorithms</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <CardDescription>
                                Our marketplace allows developers to monetize their trading strategies. Whether you’re a
                                beginner or an expert, our platform offers an opportunity to showcase your work and
                                generate revenue.
                            </CardDescription>
                        </CardContent>
                    </Card>
                </div>
            </div>

            <div className="text-center mt-12">
                <Button
                    variant={"default"}
                    color="blue"
                    size="lg"
                    className="py-3 px-6 rounded-full text-white font-semibold hover:bg-blue-700"
                    onClick={() => window.location.href = '/join'}  // Navigate to your registration or join page
                >
                    Join Our Community
                </Button>
            </div>
        </div>
    );
}
