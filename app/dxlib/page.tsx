import React from 'react';
import Link from "next/link";

const DocsPage = () => {
    return (
        <div className="flex flex-col min-h-screen">
            <header className="bg-background border-b">
                <div className="container mx-auto px-4 py-6 flex justify-between items-center">
                    <h1 className="text-2xl font-bold">divergex</h1>
                    <nav>
                        <ul className="flex space-x-4">
                            <li><Link href={"/#products"}
                                      className="text-muted-foreground hover:text-primary">Projects</Link></li>
                            <li><Link href={"/#docs"}
                                      className="text-muted-foreground hover:text-primary">Documentation</Link></li>
                        </ul>
                    </nav>
                </div>
            </header>

            <main className="flex-grow">
                <div style={{height: '100vh', overflow: 'hidden'}}>
                    <iframe
                        src="https://divergex.github.io/dxlib/"
                        style={{
                            width: '100%',
                            height: '100%',
                            border: 'none',
                        }}
                        title="Python Documentation"
                    />
                </div>
            </main>

            <footer className="bg-background border-t py-8">
                <div className="container mx-auto px-4 text-center text-muted-foreground">
                    <p>&copy; 2025 divergex. All rights reserved.</p>
                </div>
            </footer>
        </div>
    )
        ;
};

export default DocsPage;
