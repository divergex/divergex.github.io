import {Card, CardContent, CardDescription, CardHeader, CardTitle} from "@/components/ui/card";
import React, {ReactNode} from "react";
import {Prism as SyntaxHighlighter} from "react-syntax-highlighter";
import {darcula} from "react-syntax-highlighter/dist/esm/styles/prism";


interface CodeCardProps {
    icon: ReactNode;        // Type for React elements like icons
    title: string;          // Title is a string
    description: string;    // Description is a string
    codeSnippet: string;    // Code snippet is a string
    language: string;       // Language is a string
    link?: string;          // Link is an optional string
}


interface CodeSnippetProps {
    codeSnippet: string;
    language?: string;
}


export default function CodeCard({
                                     icon, title, description, codeSnippet, language, link
                                 }: CodeCardProps) {
    return (
        <Card className="overflow-hidden bg-black">
            <CardHeader>
                <div className="flex space-x-4">
                    <div className="mb-4">{icon}</div>
                    <CardTitle className="text-center py-3 text-white">{title}</CardTitle>
                </div>
                {link && (
                    <div className="mt-4">
                        <a
                            href={link}
                            rel="noopener noreferrer"
                            className="text-blue-500 hover:text-blue-700 underline"
                        >
                            Learn More
                        </a>
                    </div>
                )}
            </CardHeader>
            <CardContent>
                <CardDescription className="mb-4">{description}</CardDescription>
                <CodeSnippet codeSnippet={codeSnippet} language={language}/>
            </CardContent>
        </Card>
    );
}


const CodeSnippet = ({codeSnippet, language = 'javascript'}: CodeSnippetProps) => {
    return (
        <div className="flex flex-col items-center w-full">
            <div className="flex items-center space-x-2 bg-gray-900 p-2 rounded-t-lg w-full max-w-3xl">
                <span className="w-3.5 h-3.5 rounded-full bg-red-500"></span>
                <span className="w-3.5 h-3.5 rounded-full bg-yellow-400"></span>
                <span className="w-3.5 h-3.5 rounded-full bg-green-500"></span>
            </div>
            <SyntaxHighlighter language={language} style={darcula}
                               className="w-full max-w-3xl p-5 rounded-lg !rounded-t-none !mt-0 overflow-x-auto bg-gray-900">
                {codeSnippet}
            </SyntaxHighlighter>
        </div>
    );
};