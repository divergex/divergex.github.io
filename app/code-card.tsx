import {Card, CardContent, CardDescription, CardHeader, CardTitle} from "@/components/ui/card";
import React, {ReactNode} from "react";
import {Prism as SyntaxHighlighter} from "react-syntax-highlighter";
import {darcula} from "react-syntax-highlighter/dist/esm/styles/prism";

// declare module "react-syntax-highlighter/dist/esm/styles/prism/darcula" {
//     const style: { [key: string]: React.CSSProperties };
//     export default style;
// }


interface CodeCardProps {
    icon: ReactNode,
    title: string,
    description: string,
    codeSnippet: string,
    language: string,
    link?: string,
    className?: string,
    style?: { [key: string]: React.CSSProperties };
}

interface CodeSnippetProps {
    codeSnippet: string;
    language?: string;
    style?: { [key: string]: React.CSSProperties };
}

export default function CodeCard({
                                     icon,
                                     title,
                                     description,
                                     codeSnippet,
                                     language,
                                     link,
                                     className,
                                     style
                                 }: CodeCardProps) {
    return (
        <Card className={`${className} overflow-hidden bg-black`}>
            <CardHeader>
                <div className="flex justify-between space-x-4">
                    <div className={"flex"}>
                        <div className="mb-4">{icon}</div>
                        <CardTitle className="text-center ml-4 py-1 text-white text-2xl">{title}</CardTitle>
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
                </div>
            </CardHeader>
            <CardContent>
                <CardDescription className="mb-4 w-5/6 text-center mx-auto text-white">{description}</CardDescription>
                <CodeSnippet codeSnippet={codeSnippet} language={language} style={style}/>
            </CardContent>
        </Card>
    );
}

const CodeSnippet = ({codeSnippet, language = 'javascript', style = darcula}: CodeSnippetProps) => {
    return (
        <div className="flex flex-col items-center w-full">
            <div className="flex items-center space-x-2 bg-gray-900 p-2 rounded-t-lg w-full max-w-3xl">
                <span className="w-3.5 h-3.5 rounded-full bg-red-500"></span>
                <span className="w-3.5 h-3.5 rounded-full bg-yellow-400"></span>
                <span className="w-3.5 h-3.5 rounded-full bg-green-500"></span>
            </div>
            <SyntaxHighlighter language={language} style={style}
                               className="w-full max-w-3xl p-5 rounded-lg !rounded-t-none !mt-0 overflow-x-auto bg-gray-900">
                {codeSnippet}
            </SyntaxHighlighter>
        </div>
    );
};